import torch

from infer import ContinuousBatcher, Request, generate_kv, generate_naive, generate_speculative, static_batch_steps
from infer.data import TOK
from infer.model import GPTConfig, TinyGPT


def models():
    torch.manual_seed(0)
    t = TinyGPT(GPTConfig(TOK.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32)).eval()
    torch.manual_seed(1)
    d = TinyGPT(GPTConfig(TOK.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=16)).eval()
    return t, d


P = TOK.encode("the robot ")


def test_kv_cache_matches_naive():
    t, _ = models()
    assert generate_kv(t, P, 20) == generate_naive(t, P, 20)


def test_continuous_batching_matches_individual_generation():
    t, _ = models()
    cb = ContinuousBatcher(t, max_batch=3)
    prompts = [TOK.encode(s) for s in ["the robot ", "a drone scans ", "an agent "]] * 2
    lens = [5, 12, 3, 9, 1, 7]
    for i, (p, n) in enumerate(zip(prompts, lens)):
        cb.submit(Request(i, p, n))
    out = cb.run_all()
    for i, (p, n) in enumerate(zip(prompts, lens)):
        assert out[i] == generate_kv(t, p, n)


def test_speculative_is_exact():
    t, d = models()
    out, stats = generate_speculative(t, d, P, 25, k=3)
    assert out == generate_kv(t, P, 25)
    assert stats["target_calls"] <= 25


def test_static_batch_step_count():
    assert static_batch_steps([1, 10, 2, 3], 2) == 10 + 3


def test_http_server_roundtrip():
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from infer.server import InferenceService, make_handler

    t, _ = models()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(InferenceService(t)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/generate",
                                 data=json.dumps({"prompt": "the robot ", "max_new_tokens": 6}).encode())
    body = json.loads(urllib.request.urlopen(req, timeout=10).read())
    srv.shutdown()
    assert body["completion"] == TOK.decode(generate_kv(t, P, 6))
