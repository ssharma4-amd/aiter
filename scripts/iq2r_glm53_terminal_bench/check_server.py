"""Post-start gate for every server: log scan + sanity prompts. Exit 1 on any failure.

usage: python3 check_server.py LOG PORT [--timeout 5400]
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BAD = (
    "silently dropped",
    "did not receive every routed expert",
    "ATOM_LOADER_STRICT_COVERAGE=false",
    "Traceback (most recent call last)",
)
SANITY = [
    ("What is 17 * 23? Reply with only the number.", r"\b391\b"),
    ("What is the capital of France? Reply with one word.", r"Paris"),
    ("Spell the word 'banana' backwards. Reply with only the result.", r"ananab"),
]


def post(port, body, timeout=900):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("port", type=int)
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5)
            break
        except OSError:  # not listening yet (URLError, refused, reset, timeout)
            text = Path(a.log).read_text(errors="replace")
            for b in BAD:
                if b in text:
                    print(f"FAIL: log contains {b!r} before ready")
                    sys.exit(1)
            if time.time() - t0 > a.timeout:
                print("FAIL: server not ready in time")
                sys.exit(1)
            time.sleep(20)
    print(f"ready after {time.time() - t0:.0f}s")
    text = Path(a.log).read_text(errors="replace")
    ok = True
    for b in BAD:
        n = text.count(b)
        print(f"log check {b!r}: {n} occurrence(s)")
        ok &= n == 0
    for q, pat in SANITY:
        r = post(
            a.port,
            {
                "model": "zai-org/GLM-5.3",
                "messages": [{"role": "user", "content": q}],
                "temperature": 0,
                "max_tokens": 4096,
            },
        )
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        good = bool(re.search(pat, content))
        ok &= good
        print(
            f"sanity {'PASS' if good else 'FAIL'}: {q!r} -> {content.strip()[:120]!r} "
            f"(reasoning {len(msg.get('reasoning_content') or '')} chars, {r['usage']['completion_tokens']} tok)"
        )
    print("SERVER CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
