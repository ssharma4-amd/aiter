"""OpenAI-compatible thinking-budget proxy for agent runs (the one the GLM-5.3 IQ2R SWE-bench and Terminal-Bench
runs use). --log FILE writes one JSON line per chat request with the raw upstream reply/replies.

POST /v1/chat/completions is forwarded to the upstream ATOM server with max_tokens capped at
--budget. If the reply hit the cap without a tool call (still thinking), the proxy renders the same
conversation with the GLM-5.3 chat template, appends the partial reasoning + "</think>", and continues
through upstream /v1/completions (--answer-tokens) so the model must produce its action. GLM
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` blocks in that text are
returned as OpenAI tool_calls. Everything else is passed through unchanged.

usage: python3 budget_proxy.py --upstream http://127.0.0.1:8011 --port 9011 --budget 16384 [--log FILE]
"""

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from transformers import AutoTokenizer

TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
STATS = {
    "requests": 0,
    "forced": 0,
    "forced_with_tool": 0,
    "tool_forced": 0,
    "upstream_errors": 0,
}
LOCK = threading.Lock()
LOGF = None
INJECT = {}


def logrec(rec):
    if LOGF is None:
        return
    with LOCK:
        LOGF.write(json.dumps(rec, ensure_ascii=False) + "\n")
        LOGF.flush()


def post(url, body, timeout=7200):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def repair(r):
    """--repair: make the reply's tool calls match what llama.cpp's grammar-constrained GLM tool calling can emit
    (exactly `bash` with one `command` string). Drops calls with no arguments, renames a single mis-named argument to
    "command", and if no valid call remains, rebuilds one from <arg_value>...</arg_value> markup left in the content.
    Returns a list of repair tags (empty = untouched)."""
    msg = r["choices"][0]["message"]
    tags, good = [], []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            a = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            a = None
        if isinstance(a, dict) and "command" in a and fn.get("name") == "bash":
            good.append(tc)
        elif (
            isinstance(a, dict)
            and len(a) == 1
            and isinstance(next(iter(a.values())), str)
        ):
            fn["name"], fn["arguments"] = "bash", json.dumps(
                {"command": next(iter(a.values()))}, ensure_ascii=False
            )
            good.append(tc)
            tags.append("renamed_arg")
        else:
            tags.append("dropped_call")
    if not good:
        text = msg.get("content") or ""
        vals = re.findall(r"<arg_value>(.*?)</arg_value>", text, re.DOTALL)
        if vals:
            good = [
                {
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": vals[0]}, ensure_ascii=False
                        ),
                    },
                }
            ]
            msg["content"] = re.sub(
                r"<tool_call>.*?(</tool_call>|$)|<arg_key>.*?</arg_key>|<arg_value>.*?</arg_value>",
                "",
                text,
                flags=re.DOTALL,
            ).strip()
            tags.append("from_markup")
    if tags:
        msg["tool_calls"] = good or None
        if good:
            r["choices"][0]["finish_reason"] = "tool_calls"
    return tags


def template_messages(messages):
    """OpenAI messages -> what the GLM template expects (tool-call arguments as dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"command": args}
                tcs.append(
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "name": fn.get("name"),
                        "function": {"name": fn.get("name"), "arguments": args},
                        "arguments": args,
                    }
                )
            m["tool_calls"] = tcs
        out.append(m)
    return out


def parse_tool_calls(text):
    calls = []
    for block in TOOL_RE.findall(text):
        name = block.split("<arg_key>")[0].strip()
        args = {k.strip(): v for k, v in ARG_RE.findall(block)}
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    content = TOOL_RE.sub("", text).strip()
    return content, calls


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/stats"):
            return self._send(200, STATS)
        try:
            with urllib.request.urlopen(A.upstream + self.path, timeout=60) as r:
                return self._send(r.status, json.load(r))
        except (OSError, ValueError) as e:
            return self._send(502, {"error": str(e)})

    def do_POST(self):
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        )
        try:
            return self._post(body)
        except (
            urllib.error.HTTPError
        ) as e:  # pass upstream errors (e.g. context length 400) through unchanged
            with LOCK:
                STATS["upstream_errors"] += 1
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _post(self, body):
        if not self.path.startswith("/v1/chat/completions"):
            return self._send(200, post(A.upstream + self.path, body))
        with LOCK:
            STATS["requests"] += 1
        want = body.get("max_tokens") or 32768
        body = dict(body, **INJECT)
        b1 = dict(body, max_tokens=min(want, A.budget), stream=False)
        r = post(A.upstream + "/v1/chat/completions", b1)
        ch = r["choices"][0]
        msg = ch["message"]
        rec = {
            "t": time.time(),
            "n_msgs": len(body["messages"]),
            "body": body,
            "req_params": {
                k: body.get(k)
                for k in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "max_tokens",
                    "seed",
                )
            },
            "first": r,
        }
        if ch.get("finish_reason") != "length" or msg.get("tool_calls"):
            rec["path"] = "pass"
            if A.repair:
                rec["repair"] = repair(r)
            logrec(rec)
            return self._send(200, r)
        # Budget exhausted while still thinking: close the think block and force the action.
        prompt = TOK.apply_chat_template(
            template_messages(body["messages"]),
            tools=body.get("tools"),
            add_generation_prompt=True,
            tokenize=False,
        )
        if not prompt.endswith("<think>"):
            prompt += "<think>"
        text = (
            prompt
            + (msg.get("reasoning_content") or "")
            + "\n</think>\n"
            + (msg.get("content") or "")
        )
        b2 = {
            "model": body.get("model"),
            "prompt": text,
            "max_tokens": A.answer_tokens,
            "temperature": body.get("temperature", 0),
            "top_p": body.get("top_p", 1.0),
            **INJECT,
        }
        r2 = post(A.upstream + "/v1/completions", b2)
        gen = r2["choices"][0].get("text", "")
        content, calls = parse_tool_calls((msg.get("content") or "") + gen)
        if not calls and body.get("tools"):
            # Still no action: force the tool call itself and let the model write the argument.
            name = body["tools"][0]["function"]["name"]
            key = next(
                iter(
                    body["tools"][0]["function"]
                    .get("parameters", {})
                    .get("properties", {"command": 0})
                )
            )
            head = f"<tool_call>{name}<arg_key>{key}</arg_key><arg_value>"
            r3 = post(
                A.upstream + "/v1/completions",
                dict(
                    b2,
                    prompt=text + gen.split("<tool_call>")[0] + "\n" + head,
                    max_tokens=2048,
                    stop=["</arg_value>"],
                ),
            )
            arg = r3["choices"][0].get("text", "").split("</arg_value>")[0]
            calls = [
                {
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({key: arg}, ensure_ascii=False),
                    },
                }
            ]
            content = TOOL_RE.sub(
                "", (msg.get("content") or "") + gen.split("<tool_call>")[0]
            ).strip()
            with LOCK:
                STATS["tool_forced"] += 1
        with LOCK:
            STATS["forced"] += 1
            STATS["forced_with_tool"] += bool(calls)
        msg = {
            "role": "assistant",
            "content": content,
            "reasoning_content": msg.get("reasoning_content"),
        }
        if calls:
            msg["tool_calls"] = calls
        r["choices"][0] = {
            "index": 0,
            "message": msg,
            "finish_reason": (
                "tool_calls" if calls else r2["choices"][0].get("finish_reason")
            ),
        }
        u = r.get("usage", {})
        r["usage"] = {
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": (u.get("completion_tokens") or 0)
            + r2["usage"]["completion_tokens"],
            "total_tokens": (u.get("total_tokens") or 0)
            + r2["usage"]["completion_tokens"],
        }
        r["budget_forced"] = True
        rec.update(path="forced", forced_text=gen, final=r)
        if A.repair:
            rec["repair"] = repair(r)
        logrec(rec)
        return self._send(200, r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8001")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--log", default=None, help="JSONL file for raw upstream replies")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="repair malformed bash tool calls (llama.cpp grammar parity)",
    )
    ap.add_argument(
        "--inject",
        default="{}",
        help='JSON merged into every upstream request, e.g. {"top_k": 40}',
    )
    ap.add_argument("--answer-tokens", type=int, default=4096)
    ap.add_argument(
        "--template", default="/models/zai-org/GLM-5.3-IQ2R/chat_template.jinja"
    )
    A = ap.parse_args()
    TOK = AutoTokenizer.from_pretrained("/models/zai-org/GLM-5.3")
    TOK.chat_template = Path(A.template).read_text()
    INJECT = json.loads(A.inject)
    if A.log:
        LOGF = open(A.log, "a")  # noqa: SIM115 (open for the life of the process)
    print(
        f"budget proxy :{A.port} -> {A.upstream}, budget {A.budget}, answer {A.answer_tokens}",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()
