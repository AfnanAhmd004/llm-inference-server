"""Decoding engine: naive vs KV-cached decoding, continuous batching and speculative decoding.

All three produce exactly the same greedy tokens as the reference; they differ only in how much
work is done per generated token. Tests check that equivalence.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .model import TinyGPT


# ----------------------------------------------------------------------------- reference & KV cache
@torch.no_grad()
def generate_naive(model: TinyGPT, prompt: list[int], n: int) -> list[int]:
    """Re-runs the full sequence for every new token: O(T^2) work per token."""
    idx = torch.tensor([prompt])
    for _ in range(n):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return idx[0, len(prompt):].tolist()


@torch.no_grad()
def generate_kv(model: TinyGPT, prompt: list[int], n: int) -> list[int]:
    """Prefill once, then feed one token at a time against cached keys and values."""
    caches = [{} for _ in model.blocks]
    logits, _ = model(torch.tensor([prompt]), kv_caches=caches)
    out, pos = [], len(prompt)
    for _ in range(n):
        tok = int(logits[0, -1].argmax())
        out.append(tok)
        logits, _ = model(torch.tensor([[tok]]), kv_caches=caches, start_pos=pos)
        pos += 1
    return out


# ----------------------------------------------------------------------------- continuous batching
@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)
    caches: list = field(default_factory=list)
    pos: int = 0
    next_token: int | None = None

    @property
    def done(self) -> bool:
        return len(self.output) >= self.max_new_tokens


def _batched_step(model: TinyGPT, reqs: list[Request]) -> None:
    """One decode step for many sequences of different lengths: pad cached K/V and mask the padding."""
    B = len(reqs)
    tokens = torch.tensor([[r.next_token] for r in reqs])
    pos = torch.tensor([r.pos for r in reqs])
    x = model.tok(tokens) + model.pos(pos)[:, None, :]
    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        h = blk.ln1(x)
        C, nh = h.shape[-1], attn.n_head
        hd = C // nh
        q = attn.q_proj(h).view(B, 1, nh, hd).transpose(1, 2)
        k_new = attn.k_proj(h).view(B, 1, nh, hd).transpose(1, 2)
        v_new = attn.v_proj(h).view(B, 1, nh, hd).transpose(1, 2)
        ks, vs = [], []
        for b, r in enumerate(reqs):
            c = r.caches[li]
            c["k"] = torch.cat([c["k"], k_new[b : b + 1]], 2)
            c["v"] = torch.cat([c["v"], v_new[b : b + 1]], 2)
            ks.append(c["k"]); vs.append(c["v"])
        L = max(k.shape[2] for k in ks)
        K = torch.cat([F.pad(k, (0, 0, 0, L - k.shape[2])) for k in ks])
        V = torch.cat([F.pad(v, (0, 0, 0, L - v.shape[2])) for v in vs])
        valid = torch.tensor([[i < k.shape[2] for i in range(L)] for k in ks])
        att = (q @ K.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(~valid[:, None, None, :], float("-inf"))
        y = (F.softmax(att, -1) @ V).transpose(1, 2).reshape(B, 1, C)
        x = x + attn.o_proj(y)
        x = x + blk.fc_out(F.gelu(blk.fc_in(blk.ln2(x))))
    logits = model.head(model.ln_f(x))[:, -1]
    for r, lg in zip(reqs, logits):
        r.output.append(r.next_token)
        r.next_token = int(lg.argmax())
        r.pos += 1


class ContinuousBatcher:
    """Iteration-level scheduling: finished sequences leave and queued ones join after every step,
    instead of waiting for the whole batch to finish (static batching)."""

    def __init__(self, model: TinyGPT, max_batch: int = 8):
        self.model, self.max_batch = model, max_batch
        self.queue: deque[Request] = deque()
        self.active: list[Request] = []
        self.steps = 0

    def submit(self, req: Request) -> None:
        self.queue.append(req)

    @torch.no_grad()
    def _admit(self) -> None:
        while self.queue and len(self.active) < self.max_batch:
            r = self.queue.popleft()
            r.caches = [{} for _ in self.model.blocks]
            logits, _ = self.model(torch.tensor([r.prompt]), kv_caches=r.caches)  # prefill
            r.pos, r.next_token = len(r.prompt), int(logits[0, -1].argmax())
            self.active.append(r)

    @torch.no_grad()
    def step(self) -> list[Request]:
        self._admit()
        if not self.active:
            return []
        _batched_step(self.model, self.active)
        self.steps += 1
        finished = [r for r in self.active if r.done]
        self.active = [r for r in self.active if not r.done]
        return finished

    def run_all(self) -> dict[int, list[int]]:
        out = {}
        while self.queue or self.active:
            for r in self.step():
                out[r.id] = r.output
        return out


def static_batch_steps(lengths: list[int], max_batch: int) -> int:
    """Decode steps needed if each batch must wait for its longest member."""
    return sum(max(lengths[i : i + max_batch]) for i in range(0, len(lengths), max_batch))


# ----------------------------------------------------------------------------- speculative decoding
@torch.no_grad()
def generate_speculative(target: TinyGPT, draft: TinyGPT, prompt: list[int], n: int, k: int = 4):
    """Greedy speculative decoding: the cheap draft proposes k tokens, the target verifies them in a
    single forward pass and keeps the longest agreeing prefix plus its own next token.
    Output is identical to the target's greedy decoding."""
    seq = list(prompt)
    target_calls, accepted = 0, 0
    while len(seq) - len(prompt) < n:
        proposal = generate_kv(draft, seq, k)
        logits, _ = target(torch.tensor([seq + proposal]))
        target_calls += 1
        preds = logits[0, len(seq) - 1 :].argmax(-1).tolist()  # target's choice at each proposed position
        m = 0
        while m < k and proposal[m] == preds[m]:
            m += 1
        accepted += m
        seq += proposal[:m] + [preds[m]]
    out = seq[len(prompt) : len(prompt) + n]
    return out, {"target_calls": target_calls, "acceptance": accepted / max(1, target_calls * k)}
