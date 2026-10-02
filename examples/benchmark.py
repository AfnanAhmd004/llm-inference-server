"""Measure the effect of KV caching, continuous batching and speculative decoding."""
import random
import time

import torch

from infer import ContinuousBatcher, Request, generate_kv, generate_naive, generate_speculative, static_batch_steps
from infer.data import TOK
from infer.weights import load_or_train

torch.set_num_threads(4)
target, draft = load_or_train()
prompt = TOK.encode("the robot moves the ")
N = 96


def timed(fn):
    t0 = time.perf_counter(); out = fn(); return out, time.perf_counter() - t0


(_, t_naive), (out_kv, t_kv) = timed(lambda: generate_naive(target, prompt, N)), timed(lambda: generate_kv(target, prompt, N))
print(f"KV cache:      {N / t_naive:7.0f} tok/s naive  ->  {N / t_kv:7.0f} tok/s cached   ({t_naive / t_kv:.1f}x)")

rng = random.Random(0)
lengths = [rng.choice([8, 16, 64, 96]) for _ in range(32)]
reqs = [Request(i, TOK.encode("a drone "), n) for i, n in enumerate(lengths)]
cb = ContinuousBatcher(target, max_batch=8)
for r in reqs:
    cb.submit(r)
(_, t_cb) = timed(cb.run_all)
print(f"batching:      {static_batch_steps(lengths, 8)} decode steps static  ->  {cb.steps} continuous "
      f"({static_batch_steps(lengths, 8) / cb.steps:.2f}x fewer), {sum(lengths) / t_cb:.0f} tok/s overall")

(spec, stats), t_spec = timed(lambda: generate_speculative(target, draft, prompt, N, k=4))
assert spec == out_kv
print(f"speculative:   {stats['target_calls']} target passes for {N} tokens (vs {N}), "
      f"draft acceptance {stats['acceptance']:.0%}, identical output: True")
print("\nsample:", TOK.decode(prompt + out_kv)[:120])
