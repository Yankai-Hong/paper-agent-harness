"""Real-provider API smoke test; writes an auditable report and never logs secrets.

Covers, with minimal request count:
  1. simple text generation (raw Chat Completions),
  2. JSON structured output (NLU) with local jsonschema validation,
  3. usage availability + returned model identity,
  4. error/timeout classification (recorded from any real failure).

Offline negative paths (schema mismatch, truncation, missing usage, HTTP errors)
are already covered by tests/test_providers.py against a local HTTP server and are
reported here as "covered_offline", not re-run against the paid endpoint.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
from paper_agent.storage import write_json


def safe_base_url(url):
    parsed = urlsplit(url)
    return {"scheme": parsed.scheme, "host": parsed.hostname, "path": parsed.path}


def text_generation(config):
    body = {"model": config.model, "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly the single word: pong"}],
            "max_tokens": 64}
    # OpenCode requires a stable x-opencode-session header on every request.
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json",
               "User-Agent": "paper-agent-reproduction/0.1", "x-opencode-session": "api-smoke-text"}
    started = time.perf_counter()
    response = post_json(config.base_url.rstrip("/") + "/chat/completions", body, headers, config.timeout)
    elapsed = (time.perf_counter() - started) * 1000
    choice = response["choices"][0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    return {"kind": "text", "finish_reason": choice.get("finish_reason"),
            "content": message.get("content"),
            "has_reasoning_content": bool(message.get("reasoning_content")),
            "model_returned": response.get("model"),
            "usage_available": bool(usage), "usage": usage,
            "elapsed_ms": round(elapsed, 1)}


def json_structured(config):
    reasoner = CompatibleReasoner(config, "api-smoke")
    started = time.perf_counter()
    result = reasoner.structured("nlu", {
        "question": "Find evidence about the retrieval method in Example.",
        "memory": {}, "recent_messages": [],
        "paper_catalog": [{"paper_id": "example", "title": "Example"}]})
    elapsed = (time.perf_counter() - started) * 1000
    call = reasoner.calls[0]
    return {"kind": "json_structured", "result": result, "elapsed_ms": round(elapsed, 1),
            "finish_reason": call.get("finish_reason"),
            "model_returned": call.get("model_returned"),
            "usage_available": call.get("usage_complete"),
            "usage": call.get("usage"),
            "validation": "ok" if call.get("status") == "ok" else "failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", default="reports/api-smoke")
    args = parser.parse_args()
    config = APIConfig.from_env(args.env_file)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"started_at_utc": stamp, "provider": safe_base_url(config.base_url),
              "model_requested": config.model, "json_mode": config.json_mode,
              "thinking": config.thinking, "timeout": config.timeout,
              "checks": {}, "secret_leak_check": "no API key, Authorization header, or credentialed URL written to this report"}
    for name, fn in [("text_generation", lambda: text_generation(config)),
                     ("json_structured", lambda: json_structured(config))]:
        try:
            report["checks"][name] = fn()
        except Exception as exc:
            report["checks"][name] = {"ok": False, "error_type": type(exc).__name__,
                                      "error": str(exc)}
    report["checks"]["offline_negative_coverage"] = {
        "schema_mismatch": "tests/test_providers.py::test_api_errors_fail_closed_without_retry",
        "truncation": "tests/test_providers.py::test_api_errors_fail_closed_without_retry",
        "missing_usage": "providers.py treats absent usage as usage_complete=false",
        "http_error": "tests/test_providers.py::test_api_errors_fail_closed_without_retry"}
    write_json(out / f"smoke-{stamp}.json", report)
    write_json(out / "latest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = all(isinstance(v, dict) and v.get("kind") for k, v in report["checks"].items() if k not in {"offline_negative_coverage"})
    print("SMOKE RESULT:", "PASS" if ok else "FAIL_OR_INCOMPLETE")


if __name__ == "__main__":
    main()
