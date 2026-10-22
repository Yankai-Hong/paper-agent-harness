"""Minimal real OpenCode regression using the previous failed t3 evidence."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.memory import validate_citations
from paper_agent.storage import read_jsonl, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/acceptance/budget")
    args = p.parse_args()
    config = APIConfig.from_env(".env")
    if "opencode.ai" not in config.base_url:
        raise ValueError("Expected configured OpenCode provider")
    traces = [json.loads(f.read_text(encoding="utf-8")) for f in Path("reports/runs/e2e-d01-20261022-175339/traces/d01").glob("*.json")]
    trace = next(t for t in traces if t["status"] == "error")
    nlu = next(s["output"] for s in trace["stages"] if s["stage"] == "NLU")
    ids = next(s["output"]["evidence_ids"] for s in reversed(trace["stages"]) if s["stage"] == "Rerank")
    chunks = {c["chunk_id"]: c for c in read_jsonl("indexes/bge/chunks.jsonl")}
    payload = {"question": trace["question"], "resolved_question": nlu["standalone_query"], "evidence": [chunks[i] for i in ids]}
    write_json(Path(args.output)/"input.json", payload)
    results = []
    for budget, thinking in [(config.max_tokens, config.thinking), (4096, config.thinking), (8192, config.thinking), (4096, "disabled")]:
        reasoner = CompatibleReasoner(replace(config, max_tokens=budget, thinking=thinking, repair_retries=0), "budget-t3")
        record = {"max_tokens": budget, "thinking_requested": thinking, "test_type": "real_opencode_generate_component"}
        try:
            record["answer"] = reasoner.structured("generate", payload)
            validate_citations(record["answer"]["claims"], payload["evidence"])
            record["status"] = "ok"
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
        record["calls"] = reasoner.calls
        results.append(record)
        write_json(Path(args.output)/"results.json", results)
        print(json.dumps({k:v for k,v in record.items() if k not in {"calls", "answer"}}), flush=True)
        if any(c.get("http_status") in {401,403,429} for c in reasoner.calls):
            break


if __name__ == "__main__": main()
