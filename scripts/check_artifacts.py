"""Read-only current release checks; no API calls and no evidence rewriting."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from paper_agent.storage import read_jsonl, digest
from paper_agent.evaluation import retrieval_metrics

root = Path(__file__).resolve().parents[1]
current = root / "reports/final-acceptance"
documents = [root / "README.md", root / "reports/README.md", root / "reports/FINAL_ACCEPTANCE.md",
             root / "reports/final-acceptance/README.md", root / "reports/history/README.md",
             root / "scripts/README.md", *sorted((root / "docs").glob("*.md"))]
broken = []
for document in documents:
    for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        if "://" not in target and not target.startswith("#"):
            if not (document.parent / target.split("#", 1)[0]).exists():
                broken.append(f"{document.relative_to(root)} -> {target}")
assert not broken, broken
freeze = json.loads((current / "live-v2/freeze.json").read_text(encoding="utf-8"))
for name, expected in freeze["source_hashes"].items():
    assert hashlib.sha256((root / "src/paper_agent" / name).read_bytes()).hexdigest() == expected, name
for name, expected in freeze["dataset_hashes"].items():
    assert digest(read_jsonl(root / "data/eval/v0.6" / name)) == expected, name
queries = read_jsonl(root / "data/eval/v0.6/retrieval.jsonl")
verified = {}
for mode in ("bm25", "dense", "hybrid", "rerank"):
    directory = current / "retrieval-reused" / mode
    actual = retrieval_metrics(queries, read_jsonl(directory / "predictions.jsonl"))
    recorded = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    for key in ("Recall@5", "Hit@5", "errors", "missing"):
        assert actual[key] == recorded[key], (mode, key)
    verified[mode] = actual["Recall@5"]
suites = list(ET.parse(current / "final-v2-tests.xml").getroot().iter("testsuite"))
counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "errors", "failures", "skipped")}
assert counts["tests"] == 102 and counts["errors"] == counts["failures"] == 0, counts
predictions = read_jsonl(current / "live-v2/e2e/predictions.jsonl")
assert len(predictions) == len({p["turn_id"] for p in predictions}) == 120
for pred in predictions:
    path = current / "live-v2/e2e/traces" / pred["session_id"] / (pred["turn_id"] + ".json")
    trace = json.loads(path.read_text(encoding="utf-8"))
    assert trace["mode"] == "rerank" and trace["memory_backend"] == "three_layer"
    assert trace["commit_status"] == "committed"
print(json.dumps({"status": "passed", "documents": len(documents), "frozen_tests": counts,
                  "retrieval_recomputed": verified, "e2e_traces": 120,
                  "source_and_dataset_hashes": "matched", "api_calls": 0}, ensure_ascii=False))
