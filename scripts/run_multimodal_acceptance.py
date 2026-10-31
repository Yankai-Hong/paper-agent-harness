"""Run the fixed local multimodal cases under one durable 60-request budget.

This performs real paid requests. It requires prepared local indexes and models,
creates a fresh output directory, and never resumes an interrupted run. Before
running again, account for every reservation in the previous output manually.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def retrieval_from_trace(trace_path, catalog):
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    stages = [s for s in trace.get("stages", []) if s.get("stage") == "Rerank" and s.get("status") == "ok"]
    refs = stages[-1].get("output", {}).get("ranked_refs", []) if stages else []
    fields = ("paper_id", "chunk_id", "evidence_type", "page", "bbox", "figure_number", "image_path")
    hits = [{k: catalog[key][k] for k in fields if k in catalog[key]}
            for ref in refs[:5] if (key := (ref["paper_id"], ref["chunk_id"])) in catalog]
    return bool(stages), hits if stages else None


class BudgetLedger:
    """Single-use task-wide marker and durable pre-request reservations."""
    def __init__(self, path, limit=60, followup=False, review_followup=False, diagnostic_followup=False, linked_followup=False, language_followup=False):
        self.path, self.limit, self.records = Path(path), limit, []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if sum(map(bool, (followup, review_followup, diagnostic_followup, linked_followup, language_followup))) > 1:
            raise ValueError('Choose one follow-up mode')
        if followup or review_followup or diagnostic_followup or linked_followup or language_followup:
            original = json.loads(self.path.read_text(encoding='utf-8'))
            if original.get('limit') != limit or not isinstance(original.get('records'), list) or len(original['records']) > limit:
                raise ValueError('Existing shared budget is invalid')
            if any(not isinstance(r, dict) or r.get('attempt') != n or not r.get('request_id') or r.get('status') not in {'ok', 'error', 'reserved_unknown'} for n, r in enumerate(original['records'], 1)) or len({r['request_id'] for r in original['records']}) != len(original['records']):
                raise ValueError('Existing shared budget records are invalid')
            if any(r.get('status') == 'reserved_unknown' for r in original['records']):
                raise ValueError('An earlier request has unknown completion; manual accounting required')
            # A persistent exclusive claim prevents concurrent or repeated
            # follow-ups from silently resetting the shared budget.
            marker = ('.language-t02-v1.claimed' if language_followup else
                      '.linked-t02-v1.claimed' if linked_followup else
                      '.diagnose-t02-v1.claimed' if diagnostic_followup else
                      '.review-fixes-v1.claimed' if review_followup else '.followup-v1.claimed')
            with self.path.with_suffix(marker).open('x', encoding='utf-8') as f:
                f.write('single controlled follow-up\n')
                f.flush()
                os.fsync(f.fileno())
            self.records = original['records']
            return
        # Exclusive creation prevents a new output directory from silently
        # resetting the paid request budget. Never auto-reset or resume.
        with self.path.open("x", encoding="utf-8") as f:
            json.dump({"limit": limit, "records": []}, f)
            f.flush()
            os.fsync(f.fileno())

    def reserve(self, case_id, record):
        if len(self.records) >= self.limit:
            raise RuntimeError("Shared request budget exhausted")
        self.records.append({"attempt": len(self.records)+1, "case_id": case_id,
                             "request_id": record["request_id"], "stage": record["stage"],
                             "image_count": record["image_count"], "status": "reserved_unknown"})
        self.persist()

    def update(self, record):
        slot = next((x for x in self.records if x["request_id"] == record["request_id"]), None)
        if slot is not None:
            for field in ("status", "usage_raw", "usage_complete", "elapsed_ms", "http_status", "transport_error", "error_type", "finish_reason"):
                if field in record:
                    slot[field] = record[field]
            self.persist()

    def persist(self):
        save(self.path, {"limit": self.limit, "records": self.records})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "index", "visual-index", "visual-root", "visual-python", "visual-image-model", "visual-text-model"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--cases", default="data/eval/multimodal-v1/cases.json")
    p.add_argument("--mode", choices=("bm25", "dense", "hybrid", "rerank"), default="rerank")
    p.add_argument("--encoder", default="models/bge-m3")
    p.add_argument("--reranker", default="models/bge-reranker-v2-m3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--visual-device", default="cpu")
    p.add_argument("--followup-truncation", action="store_true", help="One controlled follow-up for c02/m01 using the existing budget")
    p.add_argument("--review-fixes", action="store_true", help="One controlled follow-up for t02/c02/m01/u01 after agent review")
    p.add_argument("--diagnose-t02", action="store_true", help="One local-response-recorded t02 diagnostic using remaining budget")
    p.add_argument("--verify-linked-t02", action="store_true", help="One t02 verification of trusted linked-text expansion")
    p.add_argument("--verify-answer-language", action="store_true", help="One t02 verification of original-question response language")
    args = p.parse_args()
    from paper_agent.providers import APIConfig, CompatibleReasoner, post_json
    from paper_agent.pipeline import PaperAgentState, PaperPipeline
    from paper_agent.memory_manager import MemoryManager
    from paper_agent.observability import Exporter
    from paper_agent.retrieval import Retriever, BGEEncoder, BGEReranker
    from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever
    config = replace(APIConfig.from_env(), model="deepseek-v4.1-flash", max_tokens=1024,
                     timeout=60, max_calls=60, max_retries=0, repair_retries=0, circuit_enabled=False,
                     **({'thinking': 'disabled'} if args.followup_truncation or args.review_fixes or args.diagnose_t02 or args.verify_linked_t02 or args.verify_answer_language else {}))
    cases_path = Path(args.cases)
    cases_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    if cases_hash != "c74a7cb62105e09165fea77d2d86fc08dbe09a34fd8b991728d21ebe85f02fbf":
        raise ValueError("Fixed acceptance fixture hash mismatch")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if len(cases["cases"]) != 10 or len(cases["image_queries"]) != 3:
        raise ValueError("This runner requires the fixed 10-question, 3-image fixture")
    visual_encoder = SubprocessEncoder(args.visual_python, args.visual_image_model, args.visual_text_model, args.visual_device)
    visual = VisualRetriever(args.visual_index, args.visual_root, visual_encoder)
    encoder = BGEEncoder(args.encoder, args.device, 8) if args.mode != "bm25" else None
    reranker = BGEReranker(args.reranker, args.device, 8) if args.mode == "rerank" else None
    text_index = Retriever(args.index, encoder, reranker)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config_record = {"model": config.model, "max_tokens": config.max_tokens, "timeout_seconds": config.timeout,
                     "max_calls": config.max_calls, "max_retries": 0, "repair_retries": 0,
                     "telemetry": "disabled", "cases_sha256": cases_hash,
                     "visual_index_metadata": visual.metadata, "text_index_metadata": text_index.metadata,
                     "mode": args.mode, "device": args.device, "visual_device": args.visual_device,
                     "thinking": config.thinking, "followup_truncation": args.followup_truncation, "review_fixes": args.review_fixes, "diagnose_t02": args.diagnose_t02, "verify_linked_t02": args.verify_linked_t02, "verify_answer_language": args.verify_answer_language,
                     "scope_prefix": "Only paper {paper_id}: {question}"}
    save(out / "config.json", config_record)
    task_budget = BudgetLedger(Path(__file__).resolve().parents[1] / "data/raw/multimodal-eval/acceptance-budget-v1.json", followup=args.followup_truncation, review_followup=args.review_fixes, diagnostic_followup=args.diagnose_t02, linked_followup=args.verify_linked_t02, language_followup=args.verify_answer_language)
    ledger = task_budget.records
    save(out / "api_calls.json", ledger)
    current = {"case_id": None}
    reasoner = None
    responses = []
    def transport(url, body, headers, timeout):
        # Reserve before network I/O. A crash leaves this attempt pending and
        # consuming budget; never persist headers, body, images, or response.
        record = reasoner.calls[-1]
        task_budget.reserve(current["case_id"], record)
        save(out / "api_calls.json", ledger)
        return post_json(url, body, headers, timeout, connect_timeout=config.connect_timeout)
    class RecordedReasoner(CompatibleReasoner):
        def structured(self, stage, payload, images=None):
            before = len(self.calls)
            try:
                value = super().structured(stage, payload, images=images)
                # Keep the model's locally validated structured answer and
                # check, including claims later refused by the pipeline.
                # Provider validation has already rejected echoed image data.
                if stage in {'nlu', 'reflect', 'generate', 'verify_claims'}:
                    responses.append({'case_id':current['case_id'], 'stage':stage, 'response':value})
                    save(out / 'structured-responses.json', responses)
                return value
            finally:
                for record in self.calls[before:]:
                    task_budget.update(record)
                save(out / "api_calls.json", ledger)
    reasoner = RecordedReasoner(config, "multimodal-acceptance", transport=transport, image_root=args.visual_root)
    result = {"cases": {}}
    catalog = {(e["paper_id"], e["chunk_id"]): e for e in list(text_index.by_id.values()) + visual.evidence}
    save(out / "predictions.json", result)
    # Exercise a fixed figure question first, then retain the original order for
    # the remaining fixed cases; do not change or regenerate the fixture.
    ordered_cases = sorted(cases["cases"], key=lambda case: case["id"] != "a01")
    if args.followup_truncation:
        ordered_cases = [case for case in ordered_cases if case['id'] in {'c02', 'm01'}]
    if args.review_fixes:
        ordered_cases = [case for case in ordered_cases if case['id'] in {'t02', 'c02', 'm01', 'u01'}]
        ordered_cases.append({'id': 'review-panel', 'paper_id': '1909.06708',
                              'question': 'Figure 2 的三个注意力热图分别对应什么模型？'})
    if args.diagnose_t02 or args.verify_linked_t02 or args.verify_answer_language:
        ordered_cases = [case for case in ordered_cases if case['id'] == 't02']
    for case in ordered_cases:
        current["case_id"] = case["id"]
        start, call_start = time.monotonic(), len(ledger)
        row = {"retrieval_complete": False, "top5": None, "answer": None, "answer_correct": None, "citations": None,
               "citation_location": None, "semantic_support": None, "usage": None, "failure": None}
        if len(ledger) >= 60:
            row["failure"] = "shared_budget_exhausted"
        else:
            try:
                session = "accept-" + case["id"]
                manager = MemoryManager.local(out / "memory" / session)
                pipeline = PaperPipeline(text_index, reasoner, out / "traces" / session, out / "memory" / session,
                    mode=args.mode, archive_recall=False, decomposition=False, max_retrieval_attempts=2,
                    memory_manager=manager, exporter=Exporter(enabled=False), visual_retriever=visual,
                    visual_root=args.visual_root, visual_claim_check=args.review_fixes or args.diagnose_t02 or args.verify_linked_t02 or args.verify_answer_language)
                question = f"Only paper {case['paper_id']}: {case['question']}"
                answer = pipeline.run(question, PaperAgentState(session), turn_id=session, case_id=case["id"], run_id="multimodal-acceptance")
                row["answer"] = answer.get("answer")
                row["citations"] = [c for claim in (row["answer"] or {}).get("claims", []) for c in claim.get("citations", [])]
                row["retrieval_complete"], row["top5"] = retrieval_from_trace(answer["trace_path"], catalog)
            except Exception as exc:
                row["failure"] = type(exc).__name__
                # Retrieval can succeed before a later model request fails.
                # Preserve that ranking without treating the turn as successful.
                traces = list((out / "traces" / ("accept-" + case["id"])).rglob("accept-" + case["id"] + ".json"))
                if len(traces) == 1:
                    row["retrieval_complete"], row["top5"] = retrieval_from_trace(traces[0], catalog)
        row["elapsed_seconds"] = time.monotonic() - start
        row["calls"] = ledger[call_start:]
        if row["calls"] and all(call.get("usage_complete") and
                                isinstance(call.get("usage_raw"), dict) for call in row["calls"]):
            row["usage"] = {key: sum(call["usage_raw"].get(key, 0) for call in row["calls"])
                            for key in ("prompt_tokens", "completion_tokens")}
        result["cases"][case["id"]] = row
        save(out / "predictions.json", result)
    current["case_id"] = None
    for case in ([] if args.followup_truncation or args.review_fixes or args.diagnose_t02 or args.verify_linked_t02 or args.verify_answer_language else cases["image_queries"]):
        start = time.monotonic()
        row = {"retrieval_complete": False, "top5": None, "failure": None}
        try:
            found = visual.search_image(case["query_file"], top_k=5)
            fields = ("paper_id", "chunk_id", "evidence_type", "page", "bbox", "figure_number", "image_path")
            row["top5"] = [{k: hit[k] for k in fields if k in hit} for hit in found["hits"]]
            row["retrieval_complete"] = True
        except Exception as exc:
            row["failure"] = type(exc).__name__
        row["elapsed_seconds"] = time.monotonic() - start
        result["cases"][case["id"]] = row
        save(out / "predictions.json", result)


if __name__ == "__main__":
    main()
