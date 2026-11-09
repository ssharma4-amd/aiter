"""CPU stand-in for the ATOM server, for testing the Terminal-Bench plumbing without GPUs.

OpenAI-compatible: GET /v1/models, POST /v1/chat/completions. Every chat reply is a valid terminus-2 JSON action
(run `ls -la` and declare the task complete), with a reasoning_content like GLM-5.3's. Requests are appended to --log.

usage: python3 stub_server.py --port 8001 [--log FILE]
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTION = {
    "analysis": "stub server: plumbing test, no model behind this endpoint.",
    "plan": "List the working directory, then stop.",
    "commands": [{"keystrokes": "ls -la\n", "duration": 0.5}],
    "task_complete": True,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(
            {
                "object": "list",
                "data": [{"id": A.model, "object": "model", "max_model_len": 262144}],
            }
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if A.log:
            with open(A.log, "a") as f:
                f.write(
                    json.dumps({"t": time.time(), "path": self.path, "body": body})
                    + "\n"
                )
        n = sum(len(str(m.get("content"))) for m in body.get("messages", []))
        self._send(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": A.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(ACTION),
                            "reasoning_content": "stub reasoning",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": n // 4,
                    "completion_tokens": 64,
                    "total_tokens": n // 4 + 64,
                },
            }
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="zai-org/GLM-5.3")
    ap.add_argument("--log", default=None)
    A = ap.parse_args()
    print(f"stub server :{A.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()
