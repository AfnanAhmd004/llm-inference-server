"""A small synthetic language (a probabilistic grammar of robot/warehouse sentences) for language modelling."""
from __future__ import annotations

import random

import torch

from .model import CharTokenizer

SUBJ = ["the robot", "the drone", "a forklift", "the arm", "the rover", "an agent"]
VERB = ["moves", "lifts", "scans", "places", "inspects", "tracks", "sorts"]
ADJ = ["red", "blue", "heavy", "small", "fragile", "metal", "empty"]
OBJ = ["box", "crate", "pallet", "part", "panel", "sensor", "bin"]
PLACE = ["to the left shelf", "onto the belt", "near the dock", "into bay four", "past the gate", "by the charger"]


def sentence(rng: random.Random) -> str:
    s = f"{rng.choice(SUBJ)} {rng.choice(VERB)} the {rng.choice(ADJ)} {rng.choice(OBJ)} {rng.choice(PLACE)}"
    if rng.random() < 0.3:
        s += f" and then {rng.choice(VERB)} it"
    return s + ". "


def corpus(n_sentences: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(sentence(rng) for _ in range(n_sentences))


TOK = CharTokenizer(corpus(2000, 0) + corpus(2000, 99))


def batches(text: str, block: int, batch_size: int, rng: random.Random):
    ids = torch.tensor(TOK.encode(text))
    while True:
        ix = torch.tensor([rng.randrange(len(ids) - block - 1) for _ in range(batch_size)])
        yield torch.stack([ids[i : i + block] for i in ix]), torch.stack([ids[i + 1 : i + block + 1] for i in ix])


@torch.no_grad()
def _perplexity(model, text: str, block: int = 64) -> float:
    ids = torch.tensor(TOK.encode(text))
    n = (len(ids) - 1) // block
    x = ids[: n * block].view(n, block)
    y = ids[1 : n * block + 1].view(n, block)
    model.eval()
    _, loss = model(x, y)
    return float(torch.exp(loss))
