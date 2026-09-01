"""OpenAI-compatible local server for Wyrmling (Mamba-2). Stdlib http, no extra deps.

  python serve.py --ckpt out/wyrmling_w300m.pt --port 8080
  # point any OpenAI SDK at  http://localhost:8080/v1

GET /v1/models, POST /v1/chat/completions (stream + non-stream), GET /health.
Byte-level chat framing applied server-side. Untrained checkpoints => gibberish; the
API is what's real. (The shippable cross-platform path is GGUF via llama.cpp — see GGUF.md.)
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from data import ASSISTANT, EOT, USER
from wyrmling import build, n_params

MODEL = None
DEVICE = None
MODEL_ID = "wyrmling"


def load(ckpt_path):
    global MODEL, DEVICE, MODEL_ID
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    MODEL = build(ck["config"]).to(DEVICE)
    MODEL.load_state_dict(ck["model"])
    MODEL.eval()
    MODEL_ID = f"wyrmling-{ck['config']}"
    print(f"loaded {MODEL_ID} ({n_params(ck['config'])/1e6:.1f}M) on {DEVICE}")


def encode_prompt(messages):
    ids = []
    for m in messages:
        ids.append(USER if m.get("role") != "assistant" else ASSISTANT)
        ids.extend(str(m.get("content", "")).encode("utf-8", "replace"))
    ids.append(ASSISTANT)
    return torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)


@torch.no_grad()
def gen_bytes(messages, max_tokens, temperature, top_k):
    idx = encode_prompt(messages)
    start = idx.size(1)
    out = MODEL.generate(idx, max_new_tokens=max_tokens, do_sample=True,
                         temperature=max(temperature, 1e-6), top_k=top_k or 40,
                         eos_token_id=EOT, pad_token_id=0)
    new = [b for b in out[0].tolist()[start:] if b not in (USER, ASSISTANT, EOT)]
    return bytes(new).decode("utf-8", "replace")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "model": MODEL_ID})
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "wyrmling"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        text = gen_bytes(msgs, int(req.get("max_tokens", 256)),
                         float(req.get("temperature", 0.8)), req.get("top_k", 40))
        created = int(time.time())
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            for ch in text:
                chunk = {"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                         "created": created, "model": MODEL_ID,
                         "choices": [{"index": 0, "delta": {"content": ch}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send(200, {
            "id": f"chatcmpl-{created}", "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(text), "total_tokens": len(text)},
        })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    load(a.ckpt)
    print(f"OpenAI-compatible API on http://{a.host}:{a.port}/v1")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
