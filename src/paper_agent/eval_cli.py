"""`eval` subcommand family: unified, reproducible evaluation entry points.

Subcommands:
  eval smoke       real-provider connectivity + JSON schema smoke test
  eval retrieval   offline retrieval metrics (BM25/dense/hybrid/rerank)
  eval intent      NLU intent classification against gold labels
  eval rewrite     NLU query-rewrite diagnostics against gold labels
  eval boundary    pipeline answer/refuse/clarify decisions against gold labels
  eval memory      memory checkpoint consistency after multi-turn runs
  eval e2e         end-to-end pipeline run + optional LLM-as-Judge
  eval compare     pair saved case outcomes and cluster-bootstrap differences
  eval ablation    execute the configured matrix in independent output directories

All metrics are written to <output>/metrics.json plus per-case files; LLM calls are
recorded to <output>/api_calls.json. API keys are never persisted.
"""
import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

from .providers import APIConfig, CompatibleReasoner
from .storage import read_jsonl, write_json, write_jsonl, digest
from .evaluation import evaluate_retriever
from . import eval_tasks as tasks
from .judge import judge_answer


def _config(args):
    config = APIConfig.from_env(getattr(args, "env_file", ".env"))
    overrides = {key:getattr(args, key) for key in ("max_tokens", "thinking", "max_calls") if getattr(args, key, None) is not None}
    return replace(config, **overrides)


def _reasoner(config, session_id):
    return CompatibleReasoner(config, session_id)


def _nlu_runner(reasoner):
    def run(stage, payload):
        return reasoner.structured("nlu", payload)
    return run


def _build_retriever(args):
    from .retrieval import Retriever, BGEEncoder, BGEReranker
    mode = getattr(args, "mode", "bm25")
    encoder = BGEEncoder(args.encoder, args.device, args.batch_size) if mode != "bm25" else None
    try:
        reranker = BGEReranker(args.reranker, args.device, args.batch_size) if mode == "rerank" else None
    except Exception:
        reranker = None  # Pipeline records explicit degradation on every affected turn.
    retriever = Retriever(args.index, encoder, reranker)
    if getattr(args, "workers", 1) > 1:
        from threading import RLock
        lock = RLock()
        search = retriever.search
        def locked_search(*a, **kw):
            with lock: return search(*a, **kw)
        retriever.search = locked_search
        if reranker:
            score = reranker.score
            def locked_score(*a, **kw):
                with lock: return score(*a, **kw)
            reranker.score = locked_score
    return retriever


def _pipeline_factory(args, retriever, config, output):
    from .pipeline import PaperPipeline
    rewrite = not getattr(args, "no_rewrite", False)
    reflect = not getattr(args, "no_reflect", False)
    compact = not getattr(args, "no_compact", False)
    memory_strategy = getattr(args, "memory_strategy", "bounded")

    def factory(session_id):
        reasoner = CompatibleReasoner(config, session_id)
        return PaperPipeline(retriever, reasoner, Path(output) / "traces" / session_id,
                             Path(output) / "memory", args.mode, rewrite, reflect, compact,
                             memory_strategy, archive_recall=not getattr(args, "no_archive_recall", False),
                             max_retrieval_attempts=getattr(args, "max_retrieval_attempts", 2))
    return factory


# --- smoke ------------------------------------------------------------------

def cmd_smoke(args):
    config = _config(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from .providers import post_json
    from urllib.parse import urlsplit
    checks = {}
    # simple text generation
    try:
        body = {"model": config.model, "stream": False,
                "messages": [{"role": "user", "content": "Reply with exactly the single word: pong"}],
                "max_tokens": 64}
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json",
                   "User-Agent": "paper-agent-reproduction/0.1", "x-opencode-session": "eval-smoke"}
        started = time.perf_counter()
        resp = post_json(config.base_url.rstrip("/") + "/chat/completions", body, headers, config.timeout)
        checks["text"] = {"content": resp["choices"][0]["message"].get("content"),
                          "model": resp.get("model"), "usage": resp.get("usage") or {},
                          "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:
        checks["text"] = {"error_type": type(exc).__name__, "error": str(exc)}
    # JSON structured output
    try:
        reasoner = CompatibleReasoner(config, "eval-smoke-json")
        started = time.perf_counter()
        result = reasoner.structured("nlu", {"question": "Find evidence about retrieval in Example.",
            "memory": {}, "recent_messages": [],
            "paper_catalog": [{"paper_id": "example", "title": "Example"}]})
        call = reasoner.calls[0]
        checks["json"] = {"result": result, "model": call.get("model_returned"),
                          "usage": call.get("usage"), "usage_complete": call.get("usage_complete"),
                          "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:
        checks["json"] = {"error_type": type(exc).__name__, "error": str(exc)}
    report = {"provider": {"scheme": urlsplit(config.base_url).scheme, "host": urlsplit(config.base_url).hostname,
                           "path": urlsplit(config.base_url).path},
              "model_requested": config.model, "checks": checks}
    write_json(out / "smoke.json", report)
    return report


# --- retrieval --------------------------------------------------------------

def cmd_retrieval(args):
    from .retrieval import Retriever, BGEEncoder, BGEReranker
    mode = args.mode
    encoder = BGEEncoder(args.encoder, args.device, args.batch_size) if mode != "bm25" else None
    reranker = BGEReranker(args.reranker, args.device, args.batch_size) if mode == "rerank" else None
    retriever = Retriever(args.index, encoder, reranker)
    return evaluate_retriever(retriever, args.queries, args.output, mode)


# --- intent / rewrite -------------------------------------------------------

def _intent_body(rows, predictions, per_case):
    metrics = tasks.intent_metrics(rows, predictions)
    metrics["dataset_hash"] = digest(rows)
    return metrics, per_case


# --- boundary ---------------------------------------------------------------

def cmd_boundary(args):
    config = _config(args)
    retriever = _build_retriever(args)
    factory = _pipeline_factory(args, retriever, config, args.output)
    rows = read_jsonl(args.data)
    from concurrent.futures import ThreadPoolExecutor
    predictions, per_case = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for pred, cases in pool.map(lambda row: tasks.run_boundary(factory, [row], args.boundary_mode), rows):
            predictions.update(pred)
            per_case.extend(cases)
    metrics = tasks.boundary_metrics(rows, predictions)
    metrics["boundary_mode"] = args.boundary_mode
    metrics["dataset_hash"] = digest(rows)
    return metrics, per_case


# --- memory -----------------------------------------------------------------

def cmd_memory(args):
    config = _config(args)
    retriever = _build_retriever(args)
    factory = _pipeline_factory(args, retriever, config, args.output)
    rows = read_jsonl(args.data)
    from concurrent.futures import ThreadPoolExecutor
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows: groups[row["session_id"]].append(row)
    final_memory, per_case = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for memories, cases in pool.map(lambda rs: tasks.run_memory(factory, rs), groups.values()):
            final_memory.update(memories)
            per_case.extend(cases)
    metrics = tasks.memory_metrics(rows, final_memory)
    for case in per_case:
        checks = [c["correct"] for c in metrics["per_check"] if c["id"] == case["id"]]
        case["checkpoint_pass"] = bool(checks) and all(checks) and not case.get("error")
    metrics["dataset_hash"] = digest(rows)
    return metrics, per_case


# --- e2e + judge ------------------------------------------------------------

def cmd_e2e(args):
    config = _config(args)
    retriever = _build_retriever(args)
    factory = _pipeline_factory(args, retriever, config, args.output)
    rows = read_jsonl(args.data)
    from .pipeline import PaperAgentState
    from collections import defaultdict
    per_session = defaultdict(list)
    for row in rows:
        per_session[row["session_id"]].append(row)
    predictions, judge_calls = [], []
    chunks = {c["chunk_id"]: c for c in retriever.chunks}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=getattr(args, "workers", 1)) as pool:
        futures = [pool.submit(tasks.run_e2e_session, factory, sid, rs, args.output)
                   for sid, rs in per_session.items()]
        for future in futures:
            predictions.extend(future.result())
    if getattr(args, "judge", False):
        from .judge import judge_answer
        from .storage import read_jsonl as _rl
        evidence_index = {r["turn_id"]: r for r in rows}
        for pred in predictions:
            if pred.get("error") or not pred.get("answer"):
                continue
            claims = pred["answer"].get("claims", [])
            cited = []
            for claim in claims:
                for c in claim.get("citations", []):
                    if c.get("chunk_id") in chunks:
                        cited.append(chunks[c["chunk_id"]]["text"])
            q = evidence_index.get(pred["turn_id"], {}).get("question", "")
            jrec = judge_answer(_reasoner(config, "judge-" + pred["turn_id"]), pred["turn_id"],
                                q, pred["answer"], cited, decision=pred["answer"].get("decision"))
            judge_calls.append(jrec.as_dict())
            from .observability import attach_scores
            dimensions=jrec.judgment or {'overall':None}
            attach_scores(Path(pred['trace_path']) if pred.get('trace_path') else
                          Path(args.output)/'traces'/pred['session_id']/(pred['turn_id']+'.json'),
                          {k:v.get('score') if isinstance(v,dict) else None for k,v in dimensions.items()},
                          'same_model_judge')
    metrics = {"turns": len(rows), "completed": sum(1 for p in predictions if not p.get("error")),
               "errors": sum(1 for p in predictions if p.get("error")),
               "dataset_hash": digest(rows)}
    from .eval_protocol import success_metrics
    metrics.update(success_metrics(predictions))
    metrics["label_status"] = sorted({r.get("review_status", "unknown") for r in rows})
    if judge_calls:
        metrics["judge"] = summarize_judgments(judge_calls)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output) / "metrics.json", metrics)
    write_jsonl(Path(args.output) / "predictions.jsonl", predictions)
    if judge_calls:
        (Path(args.output) / "judge").mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output) / "judge" / "judgments.json", judge_calls)
    return metrics, predictions


def summarize_judgments(judge_calls):
    ok = [j for j in judge_calls if j.get("status") == "ok"]
    dims = ["relevance", "factual_correctness", "completeness", "evidence_support",
            "citation_correctness", "overall"]
    out = {"judged": len(judge_calls), "ok": len(ok), "errors": len(judge_calls) - len(ok)}
    for dim in dims:
        scores = [j["judgment"][dim]["score"] for j in ok if j.get("judgment") and dim in j["judgment"]]
        if scores:
            out[dim] = {"mean": sum(scores) / len(scores), "n": len(scores)}
    return out


# --- compare / ablation -----------------------------------------------------

def cmd_compare(args):
    from .eval_protocol import compare_runs
    result = compare_runs(args.run_a, args.run_b, args.metric, args.seed)
    if args.output:
        write_json(args.output, result)
    return result


ABLATIONS = [
    {"id": "retrieval-bm25", "description": "BM25 only"},
    {"id": "retrieval-dense", "description": "BGE-M3 dense only"},
    {"id": "retrieval-hybrid", "description": "dense + BM25 + RRF"},
    {"id": "retrieval-rerank", "description": "hybrid + Cross-Encoder"},
    {"id": "no-rewrite", "description": "NLU intent/scope on, raw query retrieval"},
    {"id": "no-reflect", "description": "single retrieval, no evidence reflection"},
    {"id": "no-compact", "description": "full history, no memory trimming"},
    {"id": "memory-semantic", "description": "semantic summary vs bounded trimming"},
]


def cmd_ablation(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results = []
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for experiment in config["ablations"]:
        from .storage import safe_id
        output = root / safe_id(experiment["id"])
        parser = argparse.ArgumentParser()
        add_eval_parser(parser.add_subparsers(dest="command"))
        child = parser.parse_args(["eval", *experiment["args"], "--output", str(output)])
        if child.eval_task not in {"retrieval", "e2e", "memory", "boundary", "intent", "rewrite"}:
            raise ValueError("Unsupported ablation task")
        write_json(output / "run_config.json", {"experiment": experiment, "seed": config.get("seed", 42)})
        try:
            result = dispatch(child)
            results.append({"id": experiment["id"], "status": "completed", "metrics": result})
        except Exception as exc:
            results.append({"id": experiment["id"], "status": "error", "error_type": type(exc).__name__})
        write_json(root / "matrix_results.json", results)
    return results


# --- wiring -----------------------------------------------------------------

def add_eval_parser(subparsers):
    ev = subparsers.add_parser("eval", help="unified evaluation tasks")
    sub = ev.add_subparsers(dest="eval_task", required=True)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--env-file", default=".env")
    smoke.add_argument("--output", default="reports/api-smoke")

    retrieval = sub.add_parser("retrieval")
    retrieval.add_argument("queries")
    retrieval.add_argument("--index", required=True)
    retrieval.add_argument("--output", required=True)
    retrieval.add_argument("--mode", choices=["bm25", "dense", "hybrid", "rerank"], default="bm25")
    _model_options(retrieval)

    for name in ("intent", "rewrite"):
        p = sub.add_parser(name)
        p.add_argument("data")
        p.add_argument("--output", required=True)
        p.add_argument("--env-file", default=".env")
        _api_options(p)

    for name in ("boundary", "memory", "e2e"):
        p = sub.add_parser(name)
        p.add_argument("data")
        p.add_argument("--index", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--mode", choices=["bm25", "dense", "hybrid", "rerank"], default="rerank")
        p.add_argument("--env-file", default=".env")
        _api_options(p)
        p.add_argument("--no-rewrite", action="store_true")
        p.add_argument("--no-reflect", action="store_true")
        p.add_argument("--no-compact", action="store_true")
        p.add_argument("--no-archive-recall", action="store_true")
        p.add_argument("--max-retrieval-attempts", type=int, choices=[1, 2], default=2)
        p.add_argument("--boundary-mode", choices=["replay", "context-only"], default="replay")
        p.add_argument("--memory-strategy", choices=["bounded", "semantic"], default="bounded")
        _model_options(p)
        if name == "e2e":
            p.add_argument("--judge", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--metric", default="behavior_match")
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--output")

    ablation = sub.add_parser("ablation")
    ablation.add_argument("--config", default="configs/evaluation.json")
    ablation.add_argument("--output", required=True)


def _model_options(p):
    p.add_argument("--encoder", default="models/bge-m3")
    p.add_argument("--reranker", default="models/bge-reranker-v2-m3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)


def _api_options(p):
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--max-calls", type=int)
    p.add_argument("--thinking", choices=["enabled", "disabled"])
    p.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)


def dispatch(args):
    task = args.eval_task
    if task in {"intent", "rewrite", "boundary", "memory", "e2e"}:
        import hashlib
        import os
        config = _config(args)
        effective = {k:v for k,v in vars(config).items() if k != "api_key"}
        write_json(Path(args.output)/"run_config.json", {"arguments":vars(args), "api":effective,
            "memory_backend":os.getenv("MEMORY_BACKEND", "legacy"),
            "dataset_hash":digest(read_jsonl(args.data)),
            "source_hashes": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})
    if task == "smoke":
        return cmd_smoke(args)
    if task == "retrieval":
        return cmd_retrieval(args)
    if task == "intent":
        config = _config(args)
        rows = read_jsonl(args.data)
        reasoner = _reasoner(config, "eval-intent")
        predictions, per_case = tasks.run_intent(_nlu_runner(reasoner), rows)
        metrics, _ = _intent_body(rows, predictions, per_case)
        tasks.save_run(args.output, metrics, per_case, reasoner.calls)
        return metrics
    if task == "rewrite":
        config = _config(args)
        rows = read_jsonl(args.data)
        reasoner = _reasoner(config, "eval-rewrite")
        predictions, per_case = tasks.run_rewrite(_nlu_runner(reasoner), rows)
        metrics = tasks.rewrite_metrics(rows, predictions)
        metrics["dataset_hash"] = digest(rows)
        tasks.save_run(args.output, metrics, per_case, reasoner.calls)
        return metrics
    if task == "boundary":
        metrics, per_case = cmd_boundary(args)
        tasks.save_run(args.output, metrics, per_case)
        return metrics
    if task == "memory":
        metrics, per_case = cmd_memory(args)
        tasks.save_run(args.output, metrics, per_case)
        return metrics
    if task == "e2e":
        metrics, predictions = cmd_e2e(args)
        return metrics
    if task == "compare":
        return cmd_compare(args)
    if task == "ablation":
        return cmd_ablation(args)
    raise ValueError(f"Unknown eval task: {task}")
