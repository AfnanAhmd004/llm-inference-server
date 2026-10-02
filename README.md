# llm-inference-server

The core ideas behind high-throughput LLM serving engines such as vLLM and TGI, implemented from scratch and kept readable: **KV caching**, **continuous (iteration-level) batching** and **speculative decoding**, plus a small HTTP server with a background scheduler and a Dockerfile.

Every optimisation is **exact**: tests check that each produces the same greedy tokens as the naive reference.

## Components

| | What it does | Why it matters |
|---|---|---|
| `generate_kv` | prefill once, then decode one token at a time against cached keys and values | turns per-token work from O(T²) into O(T) |
| `ContinuousBatcher` | sequences of different lengths decode together (padded K/V with masking); finished requests leave and queued ones join **after every step** | no slot waits for the longest request in its batch |
| `generate_speculative` | a small draft model proposes *k* tokens; the target verifies them in **one** forward pass and keeps the agreeing prefix plus its own next token | fewer expensive target passes with identical output |
| `server.py` | standard-library HTTP endpoint `POST /generate`, backed by the batching loop in a background thread | requests from many clients share batches |

## Run

```bash
pip install -e ".[dev]"
python examples/benchmark.py      # trains a 4-layer target and a 1-layer draft on first run (~5 min CPU)
pytest

python -m infer.server --port 8000
curl -s localhost:8000/generate -d '{"prompt": "the robot ", "max_new_tokens": 40}'

docker build -t tiny-llm-server . && docker run -p 8000:8000 tiny-llm-server
```

```
KV cache:          183 tok/s naive  ->      295 tok/s cached   (1.6x)
batching:      352 decode steps static  ->  248 continuous (1.42x fewer), 1073 tok/s overall
speculative:   22 target passes for 96 tokens (vs 96), draft acceptance 84%, identical output: True
```

The KV-cache gain grows with sequence length; at 96 tokens and a tiny model, Python overhead keeps it modest. Continuous batching removes about 30% of decode steps for a realistic mix of short and long requests. With 84% draft acceptance, speculative decoding needs 4.4× fewer target passes.

## Scope

This is a CPU-scale teaching implementation. Production engines add paged KV memory (PagedAttention), fused CUDA kernels, tensor parallelism, sampling-aware speculative acceptance and streaming responses. The scheduling and verification logic here is the same in principle.

## License

MIT
