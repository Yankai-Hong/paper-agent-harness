"""Deterministic annotation assistance; never upgrades draft labels."""
import argparse
from pathlib import Path
from paper_agent.storage import read_jsonl, write_json, digest


def audit(root):
    chunks = read_jsonl("indexes/bge/chunks.jsonl")
    papers = {c["paper_id"] for c in chunks}
    units = {c["paper_id"]+"::"+s["paragraph_id"] for c in chunks for s in c["spans"]}
    results, pending = {}, []
    for path in Path(root).glob("*.jsonl"):
        rows = read_jsonl(path)
        errors, ids = [], set()
        for row in rows:
            identifier = row.get("id", row.get("turn_id", row.get("query_id")))
            if not identifier or identifier in ids: errors.append({"id":identifier, "error":"missing_or_duplicate_id"})
            ids.add(identifier)
            if not row.get("question", row.get("query")): errors.append({"id":identifier, "error":"missing_question"})
            if row.get("review_status") == "draft_unreviewed":
                pending.append({"dataset":path.name, "id":identifier,
                    "checks_required":["question/label consistency", "paper scope", "evidence support", "contradictory annotations"],
                    "missing_standalone_reference": "expected_query" in row and not row["expected_query"]})
            for paper in row.get("source_paper_ids", row.get("paper_ids", [])) or []:
                if paper not in papers: errors.append({"id":identifier, "error":"unknown_paper", "value":paper})
            for unit in row.get("reference_units", row.get("gold_units", [])):
                if unit not in units: errors.append({"id":identifier, "error":"unknown_evidence", "value":unit})
            if "checkpoints" in row and not any(row["checkpoints"].values()):
                errors.append({"id":identifier, "error":"empty_checkpoint"})
        results[path.name] = {"n":len(rows), "hash":digest(rows), "errors":errors}
    return {"datasets":results, "human_review_queue":pending,
            "labels_promoted":0, "note":"Schema/source checks cannot establish semantic correctness or human review."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/eval/v0.3")
    p.add_argument("--output", default="reports/acceptance/data-audit.json")
    args = p.parse_args()
    result = audit(args.data)
    write_json(args.output, result)
    print({"datasets":len(result["datasets"]), "pending_review":len(result["human_review_queue"]),
           "errors":sum(len(d["errors"]) for d in result["datasets"].values())})
