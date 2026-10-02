"""Minimal HTTP inference server with a background continuous-batching loop (standard library only).

    python -m infer.server --port 8000
    curl -s localhost:8000/generate -d '{"prompt": "the robot ", "max_new_tokens": 40}'
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .data import TOK
from .engine import ContinuousBatcher, Request


class InferenceService:
    def __init__(self, model, max_batch: int = 8):
        self.batcher = ContinuousBatcher(model, max_batch)
        self.lock = threading.Lock()
        self.results: dict[int, threading.Event] = {}
        self.outputs: dict[int, list[int]] = {}
        self.ids = itertools.count()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            with self.lock:
                finished = self.batcher.step()
            for r in finished:
                self.outputs[r.id] = r.output
                self.results[r.id].set()
            if not finished and not self.batcher.active and not self.batcher.queue:
                time.sleep(0.002)

    def generate(self, prompt: str, max_new_tokens: int, timeout: float = 30.0) -> str:
        rid = next(self.ids)
        self.results[rid] = threading.Event()
        with self.lock:
            self.batcher.submit(Request(rid, TOK.encode(prompt), max_new_tokens))
        if not self.results[rid].wait(timeout):
            raise TimeoutError("generation timed out")
        return TOK.decode(self.outputs.pop(rid))


def make_handler(service: InferenceService):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                text = service.generate(body["prompt"], int(body.get("max_new_tokens", 32)))
                payload, code = {"completion": text}, 200
            except (KeyError, ValueError) as e:
                payload, code = {"error": str(e)}, 400
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main():  # pragma: no cover
    from .weights import load_or_train

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    service = InferenceService(load_or_train()[0])
    print(f"serving on :{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(service)).serve_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
