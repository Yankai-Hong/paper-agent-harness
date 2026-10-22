"""Explicit success conditions and paired statistics; never infer semantic truth."""
import json
from pathlib import Path
from .storage import read_jsonl
from .eval_tasks import bootstrap_ci
from .task_metrics import rate


def assess(row, prediction, trace):
    stages = trace.get("stages", [])
    last = {s["stage"]: s for s in stages}
    answer = prediction.get("answer") or {}
    decision = answer.get("decision")
    executed = not prediction.get("error") and trace.get("status") == "completed"
    schema = decision in {"answer", "refuse", "clarify"} and isinstance(answer.get("claims"), list)
    expected = row.get("success_conditions") or {}
    expected_decision = expected.get("decision", row.get("provisional_decision") or "answer")
    nlu = last.get("NLU", {}).get("output", {})
    evidence = last.get("Rerank", {}).get("output", {}).get("evidence_ids", [])
    citation = decision == "answer" and bool(answer.get("claims")) and all(
        c.get("citations") and all(x.get("chunk_id") in evidence for x in c["citations"])
        for c in answer.get("claims", []))
    citation = citation and last.get("Generate", {}).get("citation_validation", {}).get("source_consistent", False)
    memory = last.get("Memory", {}).get("status") == "ok"
    flow = all(last.get(s, {}).get("status") == "ok" for s in ("Guardrails", "NLU", "Generate", "Memory"))
    if decision == "answer":
        flow = flow and all(last.get(s, {}).get("status") == "ok" for s in ("Retrieve", "Rerank", "Reflect"))
    # Semantic correctness stays unknown unless independently provided.
    quality = prediction.get("reviewed_answer_correct")
    task = executed and decision == expected_decision
    if expected.get("require_semantic_correctness", True) and expected_decision == "answer":
        task = False if not task else quality
    pipeline = False if not (executed and schema and flow and memory and (citation or decision != "answer")) else task
    return {"execution_success": executed, "task_success": task, "pipeline_success": pipeline,
            "behavior_match": executed and decision == expected_decision,
            "schema_valid": schema, "stage_flow_valid": flow, "citation_structure_valid": citation if decision == "answer" else None,
            "memory_committed": memory, "nlu_correct": nlu.get("intent") == row.get("provisional_intent") if row.get("provisional_intent") else None,
            "reflect_correct": None, "retrieval_effective": bool(evidence),
            "generate_valid": schema, "degraded": trace.get("degraded", False)}


def success_metrics(predictions):
    out = {}
    for key in ("execution_success", "task_success", "pipeline_success", "behavior_match"):
        values = [p.get("assessment", {}).get(key) for p in predictions]
        out[key + "_rate"] = {**rate(sum(v is True for v in values), len(values)),
                              "unknown": sum(v is None for v in values),
                              "note": "Unknown quality is not counted as success; denominator includes every case."}
        if values:
            out[key + "_rate"]["ci"] = bootstrap_ci([float(v is True) for v in values],
                [p["session_id"] for p in predictions])
    return out


def compare_runs(path_a, path_b, metric="behavior_match", seed=42):
    def read(path):
        path = Path(path)
        root = path if path.is_dir() else path.parent
        filename = next((root / name for name in ("predictions.jsonl", "per_case.jsonl") if (root / name).exists()), None)
        if filename is None:
            raise ValueError("Comparison requires saved per-case results")
        rows = read_jsonl(filename)
        metrics_path = root / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        details = {r["query_id"]:r for r in metrics.get("per_query", [])}
        for r in rows:
            if r.get("query_id") in details:
                r.update(details[r["query_id"]])
                r["session_id"] = (r.get("paper_ids") or [r["query_id"]])[0]
        indexed = {}
        for r in rows:
            key = (r.get("session_id", ""), r.get("id", r.get("turn_id", r.get("query_id"))))
            if key in indexed or key[1] is None:
                raise ValueError("Missing or duplicate case IDs")
            indexed[key] = r
        config = root / "run_config.json"
        return indexed, json.loads(config.read_text(encoding="utf-8")) if config.exists() else {k:metrics[k] for k in ("corpus_hash", "queries_hash", "mode", "candidate_k") if k in metrics} or None
    a, ca = read(path_a)
    b, cb = read(path_b)
    common = sorted(a.keys() & b.keys())
    deltas, clusters, improved, regressed, unknown = [], [], [], [], []
    for key in common:
        def score(r):
            if r.get("error"):
                return 0
            return r.get("assessment", {}).get(metric, r.get(metric))
        x, y = score(a[key]), score(b[key])
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            unknown.append(key)
            continue
        d = float(y) - float(x)
        deltas.append(d)
        clusters.append(key[0] or a[key].get("paper_id") or key[1])
        if d > 0: improved.append(key)
        if d < 0: regressed.append(key)
    return {"metric": metric, "direction": "b-a", "common_cases": len(common),
            "paired": bootstrap_ci(deltas, clusters, seed=seed) if deltas else None,
            "missing_in_b": sorted(a.keys()-b.keys()), "missing_in_a": sorted(b.keys()-a.keys()),
            "unknown_metric_cases": unknown, "improved": improved, "regressed": regressed,
            "failed_a": [k for k,v in a.items() if v.get("error")],
            "failed_b": [k for k,v in b.items() if v.get("error")],
            "configuration_mismatch": ca != cb, "configuration_missing": ca is None or cb is None,
            "config_a": ca, "config_b": cb}
