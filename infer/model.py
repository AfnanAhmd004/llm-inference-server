"""A small decoder-only Transformer (GPT-style) for CPU-scale experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        h = self.n_head
        q = self.q_proj(x).view(B, T, h, C // h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, C // h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, C // h).transpose(1, 2)
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        Tk = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // h)
        # causal mask aligned to the end of the key sequence (works with and without a cache)
        mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
        att = att.masked_fill(~mask, float("-inf"))
        y = self.drop(F.softmax(att, dim=-1)) @ v
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc_in = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache)
        return x + self.fc_out(F.gelu(self.fc_in(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0):
        B, T = idx.shape
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if kv_caches is None else kv_caches[i])
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, greedy: bool = True, stop_id: int | None = None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            nxt = logits[:, -1].argmax(-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits[:, -1], -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and (nxt == stop_id).all():
                break
        return idx


class CharTokenizer:
    def __init__(self, chars: str):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)
