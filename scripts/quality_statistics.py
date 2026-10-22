"""Separate structural citation validation from provisional semantic judgments."""
import json
from pathlib import Path
from paper_agent.storage import read_jsonl, write_json
from paper_agent.eval_tasks import bootstrap_ci


def summarize(root):
    root = Path(root)
    path = root/"predictions.jsonl"
    if not path.exists(): path=root/"per_case.jsonl"
    if not path.exists() or not (root/"semantic-review.jsonl").exists(): return None
    predictions = read_jsonl(path)
    judgments = {r["id"]:r for r in read_jsonl(root/"semantic-review.jsonl")}
    chunks = {r["chunk_id"]:r for r in read_jsonl("indexes/bge/chunks.jsonl")}
    rows, failures = [], []
    total_citations = ids_valid = sources_valid = supported = claims_judged = 0
    support_scores, support_clusters = [], []
    for pred in predictions:
        identifier = pred.get("turn_id",pred.get("id"))
        sid = pred["session_id"]
        record = judgments.get(identifier,{})
        judgment = record.get("judgment")
        trace_path=root/"traces"/sid/(pred.get("trace_id",identifier)+".json")
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        stages={s["stage"]:s for s in trace.get("stages",[])}
        evidence=set(stages.get("Rerank",{}).get("output",{}).get("evidence_ids",[]))
        structural=True
        answer=pred.get("answer") or {}
        claims=answer.get("claims",[])
        for claim in claims:
            structural = structural and bool(claim.get("citations"))
            for cite in claim.get("citations",[]):
                total_citations += 1
                valid=cite.get("chunk_id") in evidence and cite.get("chunk_id") in chunks
                source=valid and chunks[cite["chunk_id"]]["paper_id"] == cite.get("paper_id")
                ids_valid += valid
                sources_valid += source
                structural=structural and source
        if judgment:
            supported += sum(c["supported"] for c in judgment["claims"])
            claims_judged += len(judgment["claims"])
            support_scores.extend(float(c["supported"]) for c in judgment["claims"])
            support_clusters.extend([sid]*len(judgment["claims"]))
        appropriate = bool(judgment and judgment["behavior_appropriate"])
        correct=bool(judgment and judgment["answer_correct"])
        behavior = pred.get("assessment",{}).get("behavior_match",appropriate)
        composite = (not pred.get("error") and structural and correct and appropriate and behavior
                     and stages.get("Memory",{}).get("status") == "ok")
        row={"id":identifier,"session_id":sid,"answer_correct":correct,
             "behavior_appropriate":appropriate,"provisional_pipeline_success":composite,
             "citation_structure_valid":structural,"judge_error":record.get("error"),
             "reason":judgment.get("reason") if judgment else None,
             "decision":answer.get("decision"),"degraded":trace.get("degraded",False)}
        rows.append(row)
        if not composite or not all(c["supported"] for c in (judgment or {}).get("claims",[])):
            failures.append({**row,"question":trace.get("question"),"answer":answer,
                             "claim_judgments":(judgment or {}).get("claims",[])})
    metrics={"status":"provisional_same_model_judge_and_draft_conditions_not_human_gold",
        "n":len(rows),"citation_id_validity":{"correct":ids_valid,"total":total_citations},
        "citation_source_consistency":{"correct":sources_valid,"total":total_citations},
        "citation_evidence_support":{"supported":supported,"judged":claims_judged},
        "unsupported_claim_rate":1-supported/claims_judged if claims_judged else None,
        "failure_ids":[r["id"] for r in failures],"cases":rows}
    metrics["citation_support_ci"] = bootstrap_ci(support_scores,support_clusters) if support_scores else None
    for name in ("answer_correct","behavior_appropriate","provisional_pipeline_success"):
        metrics[name]=bootstrap_ci([float(r[name]) for r in rows],[r["session_id"] for r in rows])
    metrics["judge_requests"]=sum(len(r.get("calls",[])) for r in judgments.values())
    metrics["judge_known_tokens"]=sum(sum(c.get("usage",{}).values()) for r in judgments.values() for c in r.get("calls",[]))
    metrics["judge_unknown_usage"]=sum(not c.get("usage_complete") for r in judgments.values() for c in r.get("calls",[]))
    write_json(root/"quality-statistics.json",metrics)
    write_json(root/"all-quality-failures.json",failures)
    return metrics


if __name__ == "__main__":
    for path in Path("reports/acceptance").rglob("semantic-summary.json"):
        if path.parent.name == "rewrite": continue
        metrics=summarize(path.parent)
        if metrics: print(path.parent.as_posix(),metrics["n"],len(metrics["failure_ids"]))
