"""Evaluation task runners shared by the `eval` CLI subcommands.

Dataset contracts (see data/eval/README.md):
  intent.jsonl       {id, question, history?, intent, needs_clarification?}
  query_rewrite.jsonl {id, question, history?, expected_query, required_entities, constraints?}
  retrieval.jsonl    QASPER-shaped {query_id, query, paper_ids, gold_units, gold_alternatives?}
  boundary.jsonl     {id, question, history?, decision: answer|refuse|clarify}
  dialogs.jsonl      {session_id, turn_id, turn, question, expected_behavior?}
  memory.jsonl       {id, session_id, checkpoints: {entities?, facts?}, expected}

Metrics are computed offline; LLM calls are only made by the runner functions and are
recorded per turn. No function infers semantic correctness from citation presence.
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .storage import read_jsonl, write_json, write_jsonl, digest
from .task_metrics import classification, boundary, rate


# --- generic metric helpers -------------------------------------------------

def _confusion_labels(gold):
    return sorted(set(gold.values()))


def macro_f1(gold, predicted):
    """Macro-averaged F1 over gold labels; missing/invalid predictions count as a dedicated label."""
    labels = _confusion_labels(gold)
    metrics = {}
    for label in labels:
        tp = sum(g == label and predicted.get(i) == label for i, g in gold.items())
        fp = sum(g != label and predicted.get(i) == label for i, g in gold.items())
        fn = sum(g == label and predicted.get(i) != label for i, g in gold.items())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1,
                          "support": sum(g == label for g in gold.values())}
    metrics["macro_f1"] = sum(m["f1"] for m in metrics.values()) / len(labels)
    return metrics


def bootstrap_ci(scores, clusters, seed=42, samples=2000, alpha=0.05):
    """Cluster-level paired/normal bootstrap 95% percentile interval of the mean."""
    scores = np.asarray(scores, dtype=float)
    clusters = list(clusters)
    cluster_ids = sorted(set(clusters))
    if not len(scores) or len(scores) != len(clusters) or not np.isfinite(scores).all():
        raise ValueError("Need nonempty finite scores and matching clusters")
    if samples < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap parameters")
    rows_by_cluster = {c: np.flatnonzero(np.asarray(clusters) == c).tolist() for c in cluster_ids}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        drawn = rng.choice(cluster_ids, len(cluster_ids), replace=True)
        idx = [i for c in drawn for i in rows_by_cluster[c]]
        means.append(scores[idx].mean() if idx else float("nan"))
    means = np.asarray(means)
    means = means[~np.isnan(means)]
    return {"mean": float(scores.mean()), "n": len(scores), "clusters": len(cluster_ids),
            "bootstrap_95_percentile_interval": [float(np.quantile(means, alpha / 2)),
                                                 float(np.quantile(means, 1 - alpha / 2))]}


def save_run(out_dir, metrics, per_case, api_calls=None, errors=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "per_case.jsonl", per_case)
    if api_calls is not None:
        write_json(out / "api_calls.json", api_calls)
    if errors is not None:
        write_json(out / "errors.json", errors)
    return metrics


# --- intent -----------------------------------------------------------------

def run_intent(reasoner, rows):
    """Return predicted intents and per-case records; reasoner is a callable stage runner."""
    predictions, per_case = {}, []
    for row in rows:
        row_id = row["id"]
        payload = {"question": row["question"], "memory": {},
                   "recent_messages": row.get("history", []),
                   "paper_catalog": row.get("paper_catalog", [])}
        try:
            nlu = reasoner("nlu", payload)
            predictions[row_id] = nlu["intent"]
            per_case.append({"id": row_id, "intent": nlu["intent"], "error": None,
                             "session_id": row.get("session_id", row_id), "correct": nlu["intent"] == row["intent"],
                             "standalone_query": nlu.get("standalone_query")})
        except Exception as exc:
            per_case.append({"id": row_id, "intent": None, "session_id": row.get("session_id", row_id),
                             "correct": False, "error": f"{type(exc).__name__}: {exc}"})
    return predictions, per_case


def intent_metrics(rows, predictions):
    gold = {r["id"]: r["intent"] for r in rows}
    acc = classification(gold, predictions)
    per_class = macro_f1(gold, predictions)
    return {"n": len(rows), "accuracy": acc, "macro_f1": per_class["macro_f1"],
            "per_class": {k: v for k, v in per_class.items() if k != "macro_f1"},
            "confusion": acc["confusion"]}


# --- rewrite ----------------------------------------------------------------

def run_rewrite(reasoner, rows):
    for row in rows:
        if not row.get("session_id") or not isinstance(row.get("turn"), int):
            raise ValueError("Rewrite requires explicit session_id and turn; legacy contaminated datasets are unsupported")
        previous = 0
        for message in row.get("history", []):
            if not isinstance(message, dict):
                raise ValueError("History needs message provenance")
            sid, turn = message.get("session_id"), message.get("turn")
            if sid is None or turn is None:
                # Compatibility with the frozen v0.2/v0.3 source turn-ID convention.
                try:
                    sid, number = message["turn_id"].rsplit("-t", 1)
                    turn = int(number)
                except (KeyError, ValueError):
                    raise ValueError("Missing history session/turn provenance") from None
            if sid != row["session_id"] or not previous < turn < row["turn"]:
                raise ValueError("Cross-session, future, duplicate or unordered history")
            previous = turn
    predictions, per_case = {}, []
    for row in rows:
        row_id = row["id"]
        payload = {"question": row["question"], "memory": {},
                   "recent_messages": row.get("history", []),
                   "paper_catalog": row.get("paper_catalog", [])}
        try:
            nlu = reasoner("nlu", payload)
            predictions[row_id] = nlu
            per_case.append({"id": row_id, "standalone_query": nlu.get("standalone_query"),
                             "resolved_entities": nlu.get("resolved_entities"), "paper_ids": nlu.get("paper_ids"),
                             "session_id": row["session_id"], "error": None})
        except Exception as exc:
            per_case.append({"id": row_id, "error": f"{type(exc).__name__}: {exc}"})
    return predictions, per_case


def _tokens(text):
    import re
    return set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", str(text).casefold()))


def rewrite_metrics(rows, predictions):
    entity_retention, coref_correct, semantic_kept, total = [], 0, [], 0
    for row in rows:
        row_id = row["id"]
        pred = predictions.get(row_id, {})
        if pred.get("error"):
            pred = {}  # Failed rows remain in every applicable denominator.
        required = row.get("required_entities") or []
        resolved = pred.get("resolved_entities") or {}
        aliases = row.get("required_entity_aliases", {})
        text = json.dumps([pred.get("standalone_query", ""), pred.get("paper_ids", []), resolved], ensure_ascii=False)
        def normalize(value):
            import re
            return " ".join(re.findall(r"\w+", str(value).casefold()))
        retained = [e for e in required if any(normalize(a) in normalize(text) for a in aliases.get(e, [e]))]
        if required and row.get("entity_preservation_applicable", True):
            entity_retention.append(len(retained) / len(required))
        if "expected_entity" in row:
            coref_correct += (resolved.get(row["expected_entity"]) == row.get("expected_target"))
        expected = row.get("expected_query")
        if expected:
            overlap = len(_tokens(expected) & _tokens(pred.get("standalone_query", "")))
            denom = len(_tokens(expected) | _tokens(pred.get("standalone_query", ""))) or 1
            semantic_kept.append(overlap / denom)
            total += 1
    return {"n": len(rows), "entity_retention_rate": float(np.mean(entity_retention)) if entity_retention else None,
            "entity_applicable_cases": len(entity_retention),
            "coreference_accuracy": rate(coref_correct, sum("expected_entity" in r for r in rows)),
            "token_overlap_with_expected": float(np.mean(semantic_kept)) if semantic_kept else None,
            "note": "Token overlap is a diagnostic, not a standalone semantic-preservation measure."}


# --- boundary ---------------------------------------------------------------

def run_boundary(pipeline_factory, rows, mode="replay"):
    """Isolate each case; replay user history or inject explicitly marked context."""
    if mode not in {"replay", "context-only"}:
        raise ValueError("Unknown boundary mode")
    predictions, per_case = {}, []
    for row in rows:
        row_id = row["id"]
        execution_session = row.get("session_id", row_id) + "-" + row_id
        pipeline = pipeline_factory(execution_session)
        state = pipeline_state(pipeline, execution_session)
        try:
            history = row.get("history", [])
            if mode == "context-only":
                state.messages = [dict(m) if isinstance(m, dict) else {"role": "user", "content": m} for m in history]
            else:
                for message in history:
                    if isinstance(message, str) or message.get("role") == "user":
                        pipeline.run(message if isinstance(message, str) else message["content"], state)
            result = pipeline.run(row["question"], state)
            predictions[row_id] = result["answer"]["decision"]
            per_case.append({"id": row_id, "decision": result["answer"]["decision"],
                             "correct": result["answer"]["decision"] == row.get("decision"),
                             "error": None, "mode": mode, "session_id": row.get("session_id", row_id),
                             "trace_id": result.get("trace_id")})
        except Exception as exc:
            per_case.append({"id": row_id, "mode": mode, "session_id": row.get("session_id", row_id),
                             "correct": False,
                             "decision": None, "error": f"{type(exc).__name__}: {exc}"})
    return predictions, per_case


def pipeline_state(pipeline, session_id):
    from .pipeline import PaperAgentState
    return PaperAgentState(session_id)


def run_e2e_session(factory, session_id, rows, output, provenance=None):
    from .eval_protocol import assess
    pipeline = factory(session_id)
    state = pipeline_state(pipeline, session_id)
    predictions = []
    for row in sorted(rows, key=lambda r: r["turn"]):
        pred = {"turn_id": row["turn_id"], "session_id": session_id, "turn": row["turn"],
                "answer": None, "error": None}
        try:
            result = pipeline.run(row["question"], state, turn_id=row["turn_id"],case_id=row['turn_id'],run_id=str(output))
            pred.update(result)
        except Exception as exc:
            pred["error"] = f"{type(exc).__name__}: {exc}"
        from .observability import local_trace_path
        trace_path = Path(pred['trace_path']) if pred.get('trace_path') else local_trace_path(
            pipeline.trace_dir, {'turn_id':row['turn_id'], 'session_id':session_id, 'run_id':str(output)})
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        if provenance:
            pred["evaluation_provenance"] = provenance
        if provenance and trace:
            # Local trace is authoritative; only the hashes and fixed schema are
            # eligible for the external metadata whitelist, never the raw dataset.
            trace.update(dataset_hash=provenance["dataset_hash"],
                         evaluation_config_hash=provenance["evaluation_config_hash"],
                         evaluation_schema="paper_agent_e2e_rule_v1")
            write_json(trace_path, trace)
        pred["assessment"] = assess(row, pred, trace)
        from .observability import attach_scores
        attach_scores(trace_path,pred['assessment'],'rule',pipeline.exporter)
        predictions.append(pred)
        write_jsonl(Path(output)/"sessions"/(session_id+".jsonl"), predictions)
    return predictions


def boundary_metrics(rows, predictions):
    gold = {r["id"]: r["decision"] for r in rows}
    return {"n": len(rows), **boundary(gold, predictions)}


# --- memory -----------------------------------------------------------------

def run_memory(pipeline_factory, rows):
    """Run sessions in turn order and capture memory at each checkpoint, not only at the end."""
    per_session = defaultdict(list)
    for row in rows:
        per_session[row["session_id"]].append(row)
    final_memory = {}
    per_case = []
    for session_id, session_rows in per_session.items():
        pipeline = pipeline_factory(session_id)
        state = pipeline_state(pipeline, session_id)
        for row in sorted(session_rows, key=lambda r: r.get("turn", 0)):
            try:
                result = pipeline.run(row["question"], state, turn_id=row["id"])
                final_memory[row["id"]] = {"entities": dict(state.memory.entities),
                    "long_term": pipeline.memory_manager.store.memories(session_id) if pipeline.memory_manager else [],
                    "summary": dict(state.memory.summary),
                    "legacy_fact_metrics": "unavailable; retired representation"}
                per_case.append({"id": row["id"], "session_id": session_id, "error": None,
                                 "answer": result["answer"], "trace_id": result["trace_id"]})
            except Exception as exc:
                per_case.append({"id": row.get("id"), "session_id": session_id,
                                 "error": f"{type(exc).__name__}: {exc}"})
        final_memory[session_id] = {"entities": state.memory.entities, "legacy_fact_metrics": "unavailable; retired representation",
                                    "summary": state.memory.summary}
    return final_memory, per_case


def memory_metrics(rows, final_memory):
    correct, total, per_case, unknown_checks, unknown_cases = 0, 0, [], 0, 0
    for row in rows:
        row_id = row["id"]
        observed = final_memory.get(row_id, {})
        checkpoints = row.get("checkpoints") or {}
        ok = True
        row_unknown = False
        targets = checkpoints.get("discussed_papers", [])
        for paper_id in targets:
            total += 1
            hit = paper_id in observed.get("summary", {}).get("discussed_papers", [])
            ok = ok and hit
            per_case.append({"id": row_id, "kind": "discussed_papers", "expected": paper_id, "correct": hit})
        for paper_id in checkpoints.get("fact_source_papers", []):
            if observed.get('legacy_fact_metrics'):
                unknown_checks += 1
                row_unknown = True
                per_case.append({'id':row_id,'kind':'fact_source_papers','expected':paper_id,
                                 'correct':None,'reason':'retired_memory_representation'})
                continue
            total += 1
            hit = paper_id in observed.get("fact_source_papers", [])
            ok = ok and hit
            per_case.append({"id": row_id, "kind": "fact_source_papers", "expected": paper_id, "correct": hit})
        for kind in ("entities", "facts"):
            for key, value in (checkpoints.get(kind) or {}).items():
                if kind == 'facts' and observed.get('legacy_fact_metrics'):
                    unknown_checks += 1
                    row_unknown = True
                    per_case.append({'id':row_id,'key':key,'kind':kind,'expected':value,
                                     'correct':None,'reason':'retired_memory_representation'})
                    continue
                total += 1
                got = (observed.get(kind) or {}).get(key)
                if got != value:
                    ok = False
                    per_case.append({"id": row_id, "key": key, "kind": kind,
                                     "expected": value, "observed": got, "correct": False})
                else:
                    per_case.append({"id": row_id, "key": key, "kind": kind,
                                     "expected": value, "correct": True})
        unknown_cases += row_unknown
        if ok and not row_unknown and any(checkpoints.get(k) for k in ("discussed_papers", "fact_source_papers", "entities", "facts")):
            correct += 1
    return {"checkpoint_accuracy": rate(correct, len(rows)) if total or unknown_checks else None,
            "unknown_checks":unknown_checks,"unknown_cases":unknown_cases,
            "metric_note":"unknown checks are not counted as success; checkpoint accuracy is a lower bound",
            "empty_checkpoints": sum(not any((r.get("checkpoints") or {}).get(k) for k in ("entities", "facts", "discussed_papers", "fact_source_papers")) for r in rows),
            "checked_keys": total, "per_check": per_case}
