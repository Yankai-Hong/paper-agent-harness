"""Diagnose the text-generation HTTP 400 against the configured provider.

Reads .env, sends a single non-JSON chat request, and prints the server error body
(never the Authorization header or key).
"""
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from paper_agent.providers import APIConfig

config = APIConfig.from_env(".env")
variants = [
    ("plain_no_session", {"model": config.model, "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly the single word: pong"}],
        "max_tokens": 16}),
    ("with_session_header", {"model": config.model, "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly the single word: pong"}],
        "max_tokens": 16}),
    ("larger_max_tokens", {"model": config.model, "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly the single word: pong"}],
        "max_tokens": 256}),
]
for name, body in variants:
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json",
               "User-Agent": "paper-agent-reproduction/0.1"}
    if name == "with_session_header":
        headers["x-opencode-session"] = "diag-session"
    req = Request(config.base_url.rstrip("/") + "/chat/completions",
                  json.dumps(body).encode("utf-8"), headers, method="POST")
    try:
        with build_opener().open(req, timeout=config.timeout) as resp:
            print(name, "->", resp.status, resp.read().decode("utf-8")[:500])
    except HTTPError as exc:
        print(name, "-> HTTP", exc.code, exc.read().decode("utf-8", "replace")[:800])
    except Exception as exc:
        print(name, "->", type(exc).__name__, str(exc)[:300])
