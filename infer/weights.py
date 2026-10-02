"""Train (or load cached) target and draft models on the bundled synthetic language."""
from __future__ import annotations

import os
import random

import torch

from .data import TOK, batches, corpus
from .model import GPTConfig, TinyGPT

CACHE = os.path.join(os.path.dirname(__file__), "..", "outputs", "weights.pt")


def _train(model, steps, seed=0):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    it = batches(corpus(4000, 1), model.cfg.block_size, 32, random.Random(seed))
    for _ in range(steps):
        x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def load_or_train(steps: int = 1200):
    torch.manual_seed(0)
    target = TinyGPT(GPTConfig(TOK.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128))
    draft = TinyGPT(GPTConfig(TOK.vocab_size, block_size=128, n_layer=1, n_head=2, n_embd=48))
    if os.path.exists(CACHE):
        sd = torch.load(CACHE)
        target.load_state_dict(sd["target"]); draft.load_state_dict(sd["draft"])
        return target.eval(), draft.eval()
    _train(target, steps); _train(draft, steps)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save({"target": target.state_dict(), "draft": draft.state_dict()}, CACHE)
    return target, draft
