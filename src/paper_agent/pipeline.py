"""Seven-stage domain orchestration; new code, not an upstream feature.

reasoner.structured(stage, payload) must return schema-compatible dictionaries.
Offline tests inject a scripted reasoner. No API invocation exists by default.
"""
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from pathlib import Path
import time
import uuid
import math
import copy
import json
import os

from .memory import SessionMemory, ContextArchive, validate_citations
from .storage import write_json, safe_id, digest
from .observability import invocation, Exporter
from .budgets import TurnBudget, current_budget, BudgetExceeded
from .decomposition import plan, scope_guard, exclusion_kind, exclusive_kind

STAGES = ("Guardrails", "NLU", "Retrieve", "Rerank", "Reflect", "Generate", "Memory")


@dataclass
class PaperAgentState:
    session_id: str
    messages: list = field(default_factory=list)
    turn_count: int = 0
    memory: SessionMemory = None
    committed_turns: dict = field(default_factory=dict)
    archive_ids: list = field(default_factory=list)
    memory_namespace: str | None = None

    def __post_init__(self):
        safe_id(self.session_id)
        if self.memory is None:
            self.memory = SessionMemory(self.session_id)

    @classmethod
    def load(cls, root, session_id):
        import json
        snapshot = Path(root)/f"{safe_id(session_id)}.state.json"
        if snapshot.exists():
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            if data["session_id"] != session_id:
                raise ValueError("Session identity mismatch")
            data["memory"] = SessionMemory.from_snapshot(data["memory"])
            return cls(**data)
        path = Path(root)/f"{safe_id(session_id)}.history.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(session_id, history.get("messages", []), history.get("turn_count", 0),
                   SessionMemory.load(root, session_id))

    def save(self, root):
        # A single atomic snapshot is authoritative, avoiding split history/memory commits.
        write_json(Path(root)/f"{self.session_id}.state.json", asdict(self))


class TurnTrace:
    def __init__(self, session_id, question):
        self.value = {"turn_id": uuid.uuid4().hex, "session_id": session_id, "question": question,
                      "status": "running", "stages": [], "started_at": time.time()}

    @contextmanager
    def stage(self, name, inputs=None, attempt=1):
        if current_budget.get():
            current_budget.get().remaining()
        entry = {"stage": name, "attempt": attempt, "input": inputs, "status": "running",
                 "invocation_id": uuid.uuid4().hex, "started_at": time.time()}
        token = invocation.set({k:self.value[k] for k in ('turn_id','session_id','case_id','run_id') if k in self.value} | {'invocation_id':entry['invocation_id']})
        self.value["stages"].append(entry)
        started = time.perf_counter()
        try:
            yield entry
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            entry["elapsed_ms"] = (time.perf_counter()-started)*1000
            entry['ended_at'] = time.time()
            invocation.reset(token)

    def finish(self, root):
        self.value['ended_at'] = time.time()
        visited = {s["stage"] for s in self.value["stages"]}
        for name in STAGES:
            if name not in visited:
                self.value["stages"].append({"stage": name, "status": "skipped",
                                              "reason": self.value["status"], "elapsed_ms": 0})
        try:
            from .observability import trace_mapping
            self.value['observability_mapping'] = trace_mapping(self.value)
        except Exception as exc:
            self.value['observability_mapping'] = {'status':'unavailable','error_type':type(exc).__name__}
        from .observability import local_trace_path
        path=local_trace_path(root,self.value)
        write_json(path, self.value)
        return path


class PaperPipeline:
    def __init__(self, retriever, reasoner, trace_dir, memory_dir, mode="rerank", rewrite=True, reflect=True,
                 compact=True, memory_strategy="bounded", archive_recall=True, max_retrieval_attempts=2,
                 allow_degraded=True, memory_manager=None, exporter=None, decomposition=None, budget_limits=None):
        self.retriever, self.reasoner = retriever, reasoner
        self.trace_dir, self.memory_dir = Path(trace_dir), Path(memory_dir)
        self.mode, self.rewrite, self.reflect = mode, rewrite, reflect
        self.compact = compact
        if memory_strategy not in {"bounded", "semantic"}:
            raise ValueError("Unknown memory strategy")
        self.memory_strategy = memory_strategy
        self.archive_recall = archive_recall
        if not isinstance(max_retrieval_attempts, int) or not 1 <= max_retrieval_attempts <= 3:
            raise ValueError('Retrieval attempts must be between 1 and 3')
        self.max_retrieval_attempts = max_retrieval_attempts
        self.allow_degraded = allow_degraded
        self.memory_manager = memory_manager
        self.exporter = exporter or Exporter.from_env()
        self.decomposition = (os.getenv('PAPER_DECOMPOSITION','false').lower() == 'true') if decomposition is None else bool(decomposition)
        self.budget_limits = budget_limits if budget_limits is not None else {
            name:(float if name=='seconds' else int)(os.getenv('PAPER_TURN_'+name.upper(),str(default)))
            for name,default in {'seconds':180,'model_requests':8,'subqueries':6,'candidates':90,'rerank':60}.items()}
        if memory_manager is None and os.getenv("MEMORY_BACKEND", "legacy") == "three_layer":
            from .memory_manager import MemoryManager
            self.memory_manager = MemoryManager.from_env(self.memory_dir, getattr(retriever, "encoder", None))
        elif memory_manager is None:
            from .memory_manager import MemoryManager
            self.memory_manager = MemoryManager.local(self.memory_dir)

    def run(self, question, state, turn_id=None, case_id=None, run_id=None):
        budget = TurnBudget(**self.budget_limits)
        turn_id = safe_id(turn_id or uuid.uuid4().hex)
        if self.memory_manager:
            self.memory_manager.events = []
            try:
                self.memory_manager.load_working_memory(state, bounded=self.compact)
            except Exception as exc:
                failed = TurnTrace(state.session_id, question)
                failed.value.update(turn_id=turn_id, case_id=case_id, run_id=run_id, status="error", commit_status="uncommitted",
                                    memory_events=self.memory_manager.events, error_type=type(exc).__name__)
                failed.finish(self.trace_dir)
                raise
        if turn_id in state.committed_turns:
            committed = state.committed_turns[turn_id]
            if committed["question"] != question:
                raise ValueError("turn_id already committed with a different question")
            return {**committed["result"], "replayed_commit": True}
        original = copy.deepcopy(state)
        if self.reasoner is None:
            raise RuntimeError("No reasoner configured; use offline search/eval commands")
        trace = TurnTrace(state.session_id, question)
        trace.value.update(case_id=case_id, run_id=run_id)
        if turn_id is not None:
            trace.value["turn_id"] = safe_id(turn_id)
        trace.value.update(corpus_hash=self.retriever.metadata.get("corpus_hash"), mode=self.mode,
                           rewrite=self.rewrite, reflect=self.reflect, compact=self.compact,
                           memory_strategy=self.memory_strategy,
                           memory_backend="three_layer" if self.memory_manager.episodic else "sqlite_local")
        call_start = len(getattr(self.reasoner, "calls", []))
        trace.value.update(degraded=False, degradation_reasons=[], commit_status="uncommitted")
        state.turn_count += 1
        answer = {"decision": "error", "claims": []}
        result = {}
        budget_token = current_budget.set(budget)
        try:
            with trace.stage("Guardrails", {"question": question}) as event:
                if not question.strip() or len(question) > 20000:
                    raise ValueError("Question must contain 1..20000 characters")
                if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in question):
                    raise ValueError("Question contains unsupported control characters")
                event["output"] = {"accepted": True}
            with trace.stage("NLU", {"question": question}) as event:
                recalled = self.memory_manager.retrieve_memories(state, question) if self.memory_manager and self.archive_recall else ContextArchive(self.memory_dir/"archives").search(
                    state.session_id, question, current_memory=state.memory) if self.archive_recall else []
                trace.value["archive_recall"] = recalled
                catalog = list({c["paper_id"]: {"paper_id": c["paper_id"], "title": c.get("title", "")}
                                for c in self.retriever.by_id.values()}.values())
                nlu = self.reasoner.structured("nlu", {"question": question,
                    "memory": asdict(state.memory), "recent_messages": state.messages[-8:] if self.compact else state.messages,
                    "paper_catalog": catalog, "historical_context_not_evidence": recalled,
                    "decomposition_enabled": self.decomposition,
                    "retrieval_plan_instruction": 'When enabled and multiple evidence targets are needed, optionally provide 2-3 sub_questions within parent paper_ids; otherwise omit.'})
                if nlu.get("intent") not in {"search", "fact", "explain", "compare", "out_of_scope"}:
                    raise ValueError("Invalid intent")
                if not isinstance(nlu.get("standalone_query"), str) or not nlu["standalone_query"].strip():
                    raise ValueError("Missing standalone query")
                if nlu.get("paper_ids") is not None:
                    known = {p["paper_id"] for p in catalog}
                    if not isinstance(nlu["paper_ids"], list) or any(p not in known for p in nlu["paper_ids"]):
                        raise ValueError("NLU returned unknown paper IDs")
                scope_issue = scope_guard(question, catalog, nlu.get('paper_ids'), state,
                                          nlu.get('standalone_query'),nlu.get('resolved_entities'))
                if scope_issue:
                    nlu = {**nlu, 'needs_clarification':True}
                    event['scope_guard'] = scope_issue
                event["output"] = nlu
                if self.archive_recall:
                    extra = self.memory_manager.retrieve_memories(state, nlu["standalone_query"], nlu.get("paper_ids")) if self.memory_manager else ContextArchive(self.memory_dir/"archives").search(
                        state.session_id, nlu["standalone_query"], current_memory=state.memory,
                        paper_ids=nlu.get("paper_ids"))
                    merged, seen, used = [], set(), 0
                    # The manager has ranked the resolved-query candidates across all sources.
                    # Pre-NLU scores belong to a different query and must not displace them.
                    for record in (extra if self.memory_manager else recalled + extra):
                        key = (record["kind"], record.get("turn_id"), digest(record["content"]))
                        size = len(json.dumps(record, ensure_ascii=False))
                        if key not in seen and len(merged) < 6 and used + size <= 5000:
                            merged.append(record)
                            seen.add(key)
                            used += size
                    recalled = merged
                    trace.value["archive_recall"] = recalled
            evidence = []
            decision = "refuse" if nlu["intent"] == "out_of_scope" else "clarify" if nlu.get("needs_clarification") else "search"
            constraint_reason = None
            has_data_condition = (exclusion_kind(question, catalog) == 'non_paper' or
                                  exclusive_kind(question, catalog) == 'non_paper')
            if decision == 'search' and has_data_condition and not self.reflect:
                # A keyword filter cannot prove that a paper's results satisfy
                # temporal/data conditions. Never bypass evidence reflection.
                decision = 'clarify'
                constraint_reason = 'constraint_verification_unavailable'
            query = nlu["standalone_query"] if self.rewrite else question
            # The NLU's rewritten query may omit a time/data exclusion. Preserve
            # the user's original constraint for both retrieval modes.
            if has_data_condition and question not in query:
                query = question + '\n' + query
            goals = plan(question, nlu, self.decomposition and self.rewrite)
            coverage = {g['id']:[] for g in goals}
            trace.value['retrieval_plan'] = goals
            if decision == "search":
                for attempt in range(1, self.max_retrieval_attempts + 1):
                    with trace.stage("Retrieve", {"query": query}, attempt) as event:
                        # Separate spans reflect real component work, not seven LLM calls.
                        retrieval_mode = "hybrid" if self.mode == "rerank" else self.mode
                        targets = goals if attempt == 1 else [g for g in goals if g['id'] in verdict.get('missing_goal_ids',[])]
                        if not targets:
                            targets = [{'id':'repair','query':query,'paper_ids':nlu.get('paper_ids')}]
                        combined = {}
                        allowance = min(30, budget.available('candidates'))
                        if allowance < len(targets):
                            raise BudgetExceeded('turn_candidates_exhausted')
                        for target in targets:
                            target_query = target['query'] if attempt == 1 and len(goals)>1 else query
                            if attempt > 1:
                                target_query = question+'\n'+nlu['standalone_query']+'\n'+query
                            cap = max(1, allowance//len(targets))
                            result = None
                            for route in ([retrieval_mode,'bm25','dense'] if self.allow_degraded and retrieval_mode=='hybrid' else [retrieval_mode]):
                                budget.take('subqueries')
                                try:
                                    result = self.retriever.search(target_query, mode=route, top_k=cap, candidate_k=cap, paper_ids=target['paper_ids'])
                                    budget.remaining()
                                    budget.take('candidates',len(result['hits'][:cap]))
                                    if route != retrieval_mode:
                                        event.update(degraded=True, reason='retrieval_route_failed', fallback=route)
                                    break
                                except BudgetExceeded:
                                    raise
                                except Exception:
                                    if route == 'dense' or retrieval_mode != 'hybrid' or not self.allow_degraded:
                                        raise
                            for c in result['hits'][:cap]:
                                if target['paper_ids'] is not None and c['paper_id'] not in target['paper_ids']:
                                    raise ValueError('Retriever returned out-of-scope evidence')
                                combined[(c['paper_id'],c['chunk_id'])] = c
                                ref = {'paper_id':c['paper_id'],'chunk_id':c['chunk_id']}
                                if target['id'] in coverage and ref not in coverage[target['id']]:
                                    coverage[target['id']].append(ref)
                        result = {'hits':list(combined.values()),'candidate_ids':[c['chunk_id'] for c in combined.values()]}
                        event['candidate_coverage_not_semantic_support'] = copy.deepcopy(coverage)
                        event["output"] = {"candidate_ids": result["candidate_ids"],
                                           'candidate_refs':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} for c in combined.values()]}
                    with trace.stage("Rerank", {"candidate_ids": result["candidate_ids"]}, attempt) as event:
                        candidates = result["hits"]
                        if self.mode == "rerank" and candidates:
                            try:
                                budget.take('rerank',len(candidates))
                                if self.retriever.reranker is None:
                                    raise ValueError("Cross-encoder required")
                                scores = self.retriever.reranker.score(query, [c["text"] for c in candidates])
                                if len(scores) != len(candidates) or not all(math.isfinite(s) for s in scores):
                                    raise ValueError("Invalid reranker scores")
                                candidates = [c for _, c in sorted(zip(scores, candidates),
                                              key=lambda v: (-v[0], v[1]["chunk_id"]))]
                            except BudgetExceeded:
                                raise
                            except Exception as exc:
                                if not self.allow_degraded:
                                    raise
                                event.update(degraded=True, reason="reranker_unavailable", error_type=type(exc).__name__)
                        # Retain first retrieval evidence when filling gaps on retry.
                        selected = candidates[:5]
                        if len(goals)>1:
                            selected = []
                            for goal in goals:
                                selected += next(([c] for c in candidates if {'paper_id':c['paper_id'],'chunk_id':c['chunk_id']} in coverage[goal['id']]), [])
                            selected = list({(c['paper_id'],c['chunk_id']):c for c in selected+candidates}.values())[:5]
                        evidence = list({(c['paper_id'],c['chunk_id']): c for c in evidence + selected}.values())
                        event["output"] = {"evidence_ids": [c["chunk_id"] for c in evidence]}
                    with trace.stage("Reflect", {"evidence_ids": [c["chunk_id"] for c in evidence]}, attempt) as event:
                        verdict = self.reasoner.structured("reflect", {"question": question, "evidence": evidence,
                            "resolved_question": nlu["standalone_query"],
                            "attempt": attempt, "intent": nlu["intent"],
                            "subgoals": goals, "candidate_coverage_not_semantic_support": {g:[ref for ref in refs if (ref['paper_id'],ref['chunk_id']) in {(c['paper_id'],c['chunk_id']) for c in evidence}] for g,refs in coverage.items()},
                            "historical_context_not_evidence": recalled}) if self.reflect else {"decision": "answer"}
                        decision = verdict.get("decision")
                        if decision not in {"answer", "retrieve_again", "clarify", "refuse"}:
                            raise ValueError("Invalid reflection decision")
                        if any(g not in coverage for g in verdict.get('missing_goal_ids',[])):
                            raise ValueError('Unknown reflection goal')
                        uncovered = any(not any((ref['paper_id'],ref['chunk_id']) in {(c['paper_id'],c['chunk_id']) for c in evidence} for ref in refs) for refs in coverage.values())
                        if decision == 'answer' and len(goals)>1 and (uncovered or verdict.get('missing_goal_ids') or not self.reflect):
                            decision='refuse'
                            verdict={**verdict,'decision':decision,'reason_code':'missing_target_evidence'}
                        if decision == "retrieve_again" and attempt == self.max_retrieval_attempts:
                            decision = "refuse"
                            verdict = {**verdict, "decision": decision, "reason_code": "retrieval_budget_exhausted"}
                        event["output"] = verdict
                    if decision != "retrieve_again":
                        break
                    query = verdict.get("next_query", "")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("Reflection must specify next_query")
            with trace.stage("Generate", {"decision": decision}) as event:
                if decision == "answer":
                    if not evidence:
                        decision = "refuse"
                        answer = {"decision": decision, "text": "语料中没有足够证据。", "claims": []}
                    else:
                        answer = self.reasoner.structured("generate", {"question": question,
                            "resolved_question": nlu["standalone_query"], "evidence": evidence,
                            "subgoals": goals, "candidate_coverage_not_semantic_support": coverage,
                            "historical_context_not_evidence": recalled})
                        if not answer.get("claims"):
                            raise ValueError("Answer must include claims")
                        try:
                            validate_citations(answer["claims"], evidence)
                        except ValueError:
                            event.update(degraded=True, reason="citation_validation_failed")
                            # Controlled refusal: invalid citations are never silently repaired.
                            decision = "refuse"
                            answer = {"decision": decision, "reason_code": "citation_validation_failed",
                                      "text": "当前证据无法支撑可靠引用，因此不给出结论。", "claims": []}
                        else:
                            event["citation_validation"] = {"id_valid": True, "source_consistent": True, "semantic_support": "not_evaluated"}
                            answer["decision"] = "answer"
                else:
                    answer = {"decision": decision, "claims": [], "text":
                        "请明确所指论文或比较对象。" if decision == "clarify" else "当前语料证据不足或请求超出论文库范围。"}
                    if constraint_reason:
                        answer.update(reason_code=constraint_reason,
                                      text='当前无法核实时间或数据限制，请提供可验证的条件或证据。')
                event["output"] = answer
            with trace.stage("Memory", {"decision": answer["decision"]}) as event:
                if answer["decision"] == "answer":
                    state.memory.update(nlu.get("resolved_entities", {}), [])
                    state.memory.summary["discussed_papers"] = list(dict.fromkeys(
                        state.memory.summary.get("discussed_papers", []) + [c["paper_id"] for c in evidence]))
                else:
                    state.memory.unresolved = [question]
                state.messages.extend([{"role": "user", "content": question, "turn_id": turn_id},
                                       {"role": "assistant", "content": answer, "turn_id": turn_id}])
                if self.compact and not self.memory_manager:
                    if self.memory_strategy == "semantic":
                        from .compaction import semantic_compact_state
                        try:
                            event["compaction"] = semantic_compact_state(state, self.memory_dir/"archives", self.reasoner)
                        except Exception as exc:
                            event.update(degraded=True, reason="semantic_compaction_failed")
                            # Graceful fallback: semantic summary failure must not lose the turn.
                            event["compaction_fallback"] = {"from": "semantic", "to": "bounded",
                                                            "error": f"{type(exc).__name__}: {exc}"}
                            event["compaction"] = compact_state(state, self.memory_dir/"archives")
                    else:
                        event["compaction"] = compact_state(state, self.memory_dir/"archives")
                from .observability import local_trace_path
                trace_path = str(local_trace_path(self.trace_dir,trace.value))
                state.committed_turns[turn_id] = {"question": question, "result": {"answer": answer, "trace_id": turn_id,
                                                                                   "trace_path": trace_path}}
                if self.memory_manager:
                    event["compaction"] = self.memory_manager.commit(state, question, answer, evidence, turn_id,
                        original.turn_count, self.compact, self.memory_strategy, self.reasoner)
                    event["memory_events"] = self.memory_manager.events
                    if any(e.get("degraded") for e in self.memory_manager.events):
                        event.update(degraded=True, reason="memory_projection_or_compaction_degraded")
                else:
                    state.save(self.memory_dir)
                trace.value["commit_status"] = "committed"
                event["output"] = asdict(state.memory)
            trace.value.update(status="completed", decision=answer["decision"], answer=answer)
            result.update(answer=answer, trace_id=trace.value["turn_id"], trace_path=trace_path)
            return result
        except Exception as exc:
            state.__dict__.update(original.__dict__)
            trace.value["status"] = "error"
            trace.value['error_type'] = type(exc).__name__
            raise
        finally:
            if self.memory_manager:
                trace.value["memory_events"] = self.memory_manager.events
            degraded = [s.get("reason") for s in trace.value["stages"] if s.get("degraded")]
            trace.value.update(degraded=bool(degraded), degradation_reasons=degraded)
            trace.value["api_calls"] = getattr(self.reasoner, "calls", [])[call_start:]
            trace.value['budget'] = {'used':budget.used,'elapsed_seconds':budget.clock()-budget.started,'limits':self.budget_limits}
            current_budget.reset(budget_token)
            try:
                trace.finish(self.trace_dir)
            except Exception as exc:
                # The SQLite transaction may already be durable. Report the
                # missing audit projection instead of masking a committed answer.
                if trace.value['commit_status'] == 'committed':
                    result.update(trace_status='write_failed', degraded=True)
                trace.value['trace_write_error_type'] = type(exc).__name__
            try:
                self.exporter.export(trace.value)
                self.exporter.flush()
            except Exception:
                pass  # Optional export cannot change committed business state.


def compact_state(state, archive_dir, max_facts=40, max_entities=20, max_messages=8):
    """Deterministic bounded working memory; archive evicted state, no LLM summary claim."""
    if len(state.memory.entities) <= max_entities and len(state.messages) <= max_messages:
        return {"compressed": False}
    before = len(json.dumps({"memory": asdict(state.memory), "messages": state.messages}, ensure_ascii=False))
    archive_id = ContextArchive(archive_dir).store({"memory": asdict(state.memory), "messages": state.messages,
                                                  "turn_count": state.turn_count, "commit_protocol": 1})
    state.archive_ids = list(dict.fromkeys(state.archive_ids + [archive_id]))
    state.memory.entities = dict(list(state.memory.entities.items())[-max_entities:]) if max_entities else {}
    state.messages = state.messages[-max_messages:] if max_messages else []
    after = len(json.dumps({"memory": asdict(state.memory), "messages": state.messages}, ensure_ascii=False))
    return {"compressed": True, "archive_id": archive_id, "method": "bounded_recent_state",
            "context_chars_before": before, "context_chars_after": after,
            "token_estimate_before": math.ceil(before/4), "token_estimate_after": math.ceil(after/4),
            "token_measurement": "character estimate; not API token usage",
            "retained_entities": list(state.memory.entities)}
