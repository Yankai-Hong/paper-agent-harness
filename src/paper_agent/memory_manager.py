"""Three-layer coordination; historical memory is never paper evidence."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import re
import time

from .memory import SessionMemory, ContextArchive, validate_citations
from .memory_stores import FactStore, RedisWorkingStore, ChromaEpisodes, paper_ids as decode_papers
from .storage import digest


class MemoryManager:
    def __init__(self, root, working, episodic, fallback=True):
        self.root = Path(root)
        self.facts = FactStore(self.root / "facts.sqlite3")
        self.working, self.episodic, self.fallback = working, episodic, fallback
        self.events = []

    @classmethod
    def from_env(cls, root, encoder):
        return cls(root, RedisWorkingStore(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                   namespace=os.getenv("MEMORY_REDIS_NAMESPACE", "paper-agent:working:v1"),
                   ttl=int(os.getenv("MEMORY_WORKING_TTL", "86400"))),
                   ChromaEpisodes(os.getenv("MEMORY_CHROMA_PATH", str(Path(root)/"chroma")), encoder,
                                  collection=os.getenv("MEMORY_CHROMA_COLLECTION", "paper_agent_episodes_v1"),
                                  min_similarity=float(os.getenv("MEMORY_MIN_SIMILARITY", ".35"))),
                   fallback=os.getenv("MEMORY_ALLOW_FALLBACK", "true").lower() == "true")

    def _failure(self, operation, exc, committed=False):
        # Never expose connection URLs/passwords through an exception message.
        self.events.append({"operation": operation, "degraded": True, "error_type": type(exc).__name__,
                            "recovery": "durable_sqlite_and_file", "committed": committed})
        if not self.fallback and not committed:
            raise RuntimeError(f"Memory {operation} failed ({type(exc).__name__})") from None

    def load_working_memory(self, state, bounded=True):
        durable = self.facts.snapshot(state.session_id)
        cached = None
        try:
            cached = self.working.load(state.session_id)
            if cached and not durable:
                self.events.append({"operation": "ignore_orphan_cache"})
        except Exception as exc:
            self._failure("redis_read", exc)
        if durable is None:
            # Legacy states stay readable; import unverified facts on the first successful commit.
            from .pipeline import PaperAgentState
            legacy = PaperAgentState.load(self.root, state.session_id)
            if legacy.turn_count > state.turn_count:
                state.__dict__.update(legacy.__dict__)
        else:
            # Accept only a cache projection matching the durable committed version/content.
            durable_view = {**durable, "committed_turns": {}}
            if cached == durable_view:
                durable = {**cached, "committed_turns": durable["committed_turns"]}
                self.events.append({"operation": "redis_cache_hit"})
            durable["memory"] = SessionMemory(**durable["memory"])
            state.__dict__.update(durable)
            active = self.facts.facts(state.session_id, "active")
            if bounded:
                active = active[-40:]
            state.memory.facts = {f["canonical_key"]: {"text": f["text"], "evidence_ids": f["evidence_ids"],
                "fact_id": f["fact_id"], "turn_id": f["turn_id"], "source_type": f["source_type"]} for f in active}
        self.repair_projections(state)
        return state

    def update_working_memory(self, state, committed=False):
        try:
            # Keep the unbounded idempotency ledger in SQLite/file, not Redis.
            snapshot = asdict(state)
            snapshot["committed_turns"] = {}
            snapshot["messages"] = snapshot["messages"][-8:]
            snapshot["memory"]["facts"] = dict(list(snapshot["memory"]["facts"].items())[-40:])
            self.working.save(snapshot)
        except Exception as exc:
            self._failure("redis_write", exc, committed)

    def repair_projections(self, state):
        self.check_episode_index(state.session_id)
        # Repeating an upsert after a crash is safe, including crash after index write.
        for episode in self.facts.episodes(state.session_id, pending=True, limit=32):
            try:
                self.episodic.write(episode)
                with self.facts.transaction() as db:
                    db.execute("UPDATE episodes SET indexed=1 WHERE episode_id=?", (episode["episode_id"],))
            except Exception as exc:
                self._failure("chroma_outbox", exc, committed=True)
                break
        if self.facts.snapshot(state.session_id) is not None:
            try:
                state.save(self.root)
            except Exception as exc:
                self._failure("file_projection", exc, committed=True)
            self.update_working_memory(state, committed=True)

    def check_episode_index(self, session_id):
        batch = self.facts.audit_batch(session_id)
        if not batch:
            return
        try:
            indexed = self.episodic.fingerprints(session_id, [e["episode_id"] for e in batch])
            missing = [e["episode_id"] for e in batch if indexed.get(e["episode_id"]) != digest(e)]
        except Exception as exc:
            # Retain checked records for retry; do not advance the audit cursor on failure.
            with self.facts.transaction() as db:
                db.executemany("UPDATE episodes SET indexed=0 WHERE episode_id=? AND session_id=?",
                               [(e["episode_id"], session_id) for e in batch])
            self._failure("chroma_consistency", exc, committed=True)
            return
        with self.facts.transaction() as db:
            db.executemany("UPDATE episodes SET indexed=0 WHERE episode_id=? AND session_id=?",
                           [(identifier, session_id) for identifier in missing])
            db.execute("INSERT OR REPLACE INTO episode_audit_cursor VALUES (?,?)", (session_id, batch[-1]["episode_id"]))
        if missing:
            self.events.append({"operation": "chroma_repair_queued", "episode_ids": missing})

    def extract_memories(self, question, answer, evidence, turn_id):
        candidates = []
        if answer.get("decision") == "answer":
            validate_citations(answer["claims"], evidence)
            by_id = {e["chunk_id"]: e for e in evidence}
            for claim in answer["claims"]:
                refs = claim["citations"]
                # Conservative deterministic support: complete verbatim sentence/chunk only.
                # Paraphrases remain conflict candidates, not independently verified facts.
                normalized = " ".join(claim["text"].split())
                supported = all(normalized in [" ".join(s.split()) for s in
                                [by_id[r["chunk_id"]]["text"]] +
                                re.split(r"(?<=[.!?。！？])\s+", by_id[r["chunk_id"]]["text"])] for r in refs)
                papers = sorted({r["paper_id"] for r in refs})
                candidates.append({"key": claim.get("key", claim["text"]), "text": claim["text"],
                    "source_type": "paper_evidence", "paper_ids": papers,
                    "evidence_ids": [r["chunk_id"] for r in refs], "status": "active" if supported else "conflict",
                    "reason": "verbatim_evidence_sentence" if supported else "semantic_support_unverified"})
        # Explicit user memory declarations only. No inference of preferences from an ordinary question.
        match = re.fullmatch(r"(记住|偏好|约束|更新记忆|Remember|Preference|Update memory)[:：]\s*([^=：:]+)[=：:]\s*(.+)", question, re.I | re.S)
        if match:
            candidates.append({"key": match[2].strip(), "text": match[3].strip(),
                               "explicit_update": match[1].casefold() in {"更新记忆", "update memory"},
                               "source_type": "user_explicit", "status": "active", "reason": "explicit_user_declaration"})
        event_type = "user_preference" if match else None
        for category, pattern in (
            ("research_decision", r"^(?:研究决策|实现决策|决定|决策|Decision)[:：]|^(?:我|我们)决定"),
            ("experiment_result", r"^(?:关键实验结果|实验结果|Experiment result)[:：]"),
            ("method_change", r"^(?:方法变更|Method change)[:：]"),
            ("follow_up", r"^(?:后续待办|未解决问题|TODO|Unresolved)[:：]"),
            ("research_note", r"^(?:复用笔记|研究笔记|Research note)[:：]"),
            ("user_constraint", r"^(?:研究约束|Research constraint)[:：]"),
        ):
            if re.search(pattern, question.strip(), re.I):
                event_type = category
                break
        episode = None
        if event_type:
            refs = [r for c in answer.get("claims", []) for r in c.get("citations", [])]
            episode = {"turn_id": turn_id, "content": json.dumps({"question": question, "answer": answer}, ensure_ascii=False),
                       "event_type": event_type, "retention_reason": "explicit_user_marker", "created_at": time.time(),
                       "source_type": "conversation", "paper_ids": sorted({r["paper_id"] for r in refs}),
                       "evidence_ids": sorted({r["chunk_id"] for r in refs})}
        return candidates, episode

    def update_fact_memory(self, db, session_id, turn_id, candidates):
        from .memory_stores import canonical_key
        for fact in candidates:
            if fact.get("explicit_update") and fact["source_type"] == "user_explicit":
                key = json.dumps(["user_explicit", None, canonical_key(fact["key"])], ensure_ascii=False)
                row = db.execute("SELECT fact_id FROM facts WHERE session_id=? AND canonical_key=? AND status='active'",
                                 (session_id, key)).fetchone()
                if row:
                    fact["replaces"] = row[0]
        return [self.facts.put_fact(db, session_id, turn_id, fact) for fact in candidates]

    def write_episode(self, db, session_id, episode):
        if episode is None:
            return {"action": "skip_low_value"}
        episode = {**episode, "session_id": session_id}
        episode["episode_id"] = digest([session_id, episode["turn_id"], episode["event_type"]])
        # Exact-content dedup; semantic similarity only informs recall, never deletes history.
        for row in db.execute("SELECT payload FROM episodes WHERE session_id=?", (session_id,)):
            if json.loads(row[0])["content"] == episode["content"]:
                return {"action": "duplicate"}
        db.execute("INSERT OR IGNORE INTO episodes VALUES (?,?,?,0)",
                   (episode["episode_id"], session_id, json.dumps(episode, ensure_ascii=False)))
        return {"action": "queued", "episode_id": episode["episode_id"]}

    @staticmethod
    def rank_memories(records, limit=6, max_chars=5000):
        """Compare relevance tiers and within-source ranks, never raw cosine vs term counts."""
        status_order = {"active": 3, "historical": 2, "conflict": 1, "superseded": 0}
        source_order = {"paper_evidence": 2, "user_explicit": 2, "conversation": 1}
        lanes = {}
        for record in records:
            lanes.setdefault(record["retrieval_source"], []).append(dict(record))
        ranked = []
        for lane in lanes.values():
            lane.sort(key=lambda r: (-r["relevance_score"], -status_order.get(r["status"], 0),
                                    -r.get("updated_at", r.get("created_at", 0)), r["record_id"]))
            for rank, record in enumerate(lane, 1):
                record["source_rank"] = rank
                record["rank_score"] = 1 / (60 + rank)
                ranked.append(record)
        ranked.sort(key=lambda r: (-r["relevance_tier"], -r["rank_score"],
                                  -status_order.get(r["status"], 0),
                                  -source_order.get(r["source_type"], 0),
                                  -r.get("updated_at", r.get("created_at", 0)), r["record_id"]))
        selected, seen, deferred, used, fact_count = [], set(), [], 0, 0
        # Reserve opportunities for other types without exceeding the overall budget.
        mixed = any(r["retrieval_source"] != "sqlite" and
                    len(json.dumps(r, ensure_ascii=False)) <= max_chars for r in ranked)
        fact_cap = max(1, limit // 2) if mixed else limit
        reserve = next((len(json.dumps(r, ensure_ascii=False)) for r in ranked
                        if r["retrieval_source"] != "sqlite" and
                        len(json.dumps(r, ensure_ascii=False)) <= max_chars), 0)
        other_selected = False
        for record in ranked:
            if record["record_id"] in seen:
                continue
            seen.add(record["record_id"])
            if record["retrieval_source"] == "sqlite" and fact_count >= fact_cap:
                deferred.append(record)
                continue
            size = len(json.dumps(record, ensure_ascii=False))
            if record["retrieval_source"] == "sqlite" and mixed and not other_selected and used + size + reserve > max_chars:
                deferred.append(record)
                continue
            if len(selected) < limit and used + size <= max_chars:
                selected.append(record)
                used += size
                fact_count += record["retrieval_source"] == "sqlite"
                other_selected = other_selected or record["retrieval_source"] != "sqlite"
        # Unused capacity can be reclaimed; no type reservation wastes available slots.
        for record in deferred:
            size = len(json.dumps(record, ensure_ascii=False))
            if len(selected) < limit and used + size <= max_chars:
                selected.append(record)
                used += size
        return selected

    def retrieve_memories(self, state, query, paper_ids=None, event_type=None, limit=6, max_chars=5000):
        from .corpus import lexical_tokens
        tokens = set(lexical_tokens(query))
        if not tokens or limit <= 0 or max_chars <= 0:
            return []
        scope = set(decode_papers(paper_ids))
        history = bool(re.search(r"历史|变更|之前|曾经|旧版|原来|\b(history|previous|formerly|changed)\b", query, re.I))
        records = []
        def lexical(text):
            return len(tokens & set(lexical_tokens(text))) / len(tokens)
        def add(record, source, score, score_kind, status, identity):
            if score < (.2 if score_kind == "query_coverage" else self.episodic.min_similarity):
                return
            record = {**record, "session_id": state.session_id, "record_id": identity,
                      "retrieval_source": source, "source_type": record.get("source_type", "conversation"),
                      "status": status, "relevance_score": score, "score_kind": score_kind,
                      "relevance_tier": 2 if score >= (.6 if score_kind == "query_coverage" else .65) else 1,
                      "historical_context_not_evidence": True, "is_authoritative": False,
                      "is_current_fact": source == "sqlite" and status == "active",
                      "requires_resolution": status == "conflict"}
            if status == "conflict":
                record["kind"] = "conflict"
            records.append(record)
        all_facts = self.facts.facts(state.session_id)
        by_id = {f["fact_id"]: f for f in all_facts}
        fact_signatures = {digest([f["text"], sorted(f["evidence_ids"])]) for f in all_facts}
        for fact in all_facts:
            if fact["status"] not in ({"active", "conflict", "superseded"} if history else {"active", "conflict"}):
                continue
            if scope and not scope.intersection(fact["paper_ids"]):
                continue
            text = fact["text"] + " " + json.loads(fact["canonical_key"])[-1]
            add({**fact, "kind": "fact", "content": fact["text"]}, "sqlite", lexical(text),
                "query_coverage", fact["status"], "fact:" + fact["fact_id"])
        episodes = {e["episode_id"]: e for e in self.facts.episodes(state.session_id)}
        self.check_episode_index(state.session_id)
        try:
            hits = self.episodic.search(state.session_id, query, paper_ids, event_type, max(limit * 4, 24))
        except Exception as exc:
            self._failure("chroma_read", exc)
            hits = [{"episode_id": i, "relevance_score": lexical(e["content"]),
                     "score_kind": "query_coverage"} for i, e in episodes.items()]
        for hit in hits:
            episode = episodes.get(hit["episode_id"])
            if not episode or (scope and not scope.intersection(decode_papers(episode["paper_ids"]))):
                continue
            if event_type and episode["event_type"] != event_type:
                continue
            if hit["score_kind"] != "query_coverage" and hit.get("payload_hash") != digest(episode):
                with self.facts.transaction() as db:
                    db.execute("UPDATE episodes SET indexed=0 WHERE episode_id=?", (hit["episode_id"],))
                continue  # Stale index version is never served as a current candidate.
            add({**episode, "kind": "episode", "distance": hit.get("distance"), "metric": hit.get("metric")},
                "chroma", hit["relevance_score"], hit["score_kind"], "historical",
                "event:" + state.session_id + ":" + episode["turn_id"])
        event_turns = {e["turn_id"] for e in episodes.values()}
        archives = ContextArchive(self.root/"archives").search(state.session_id, query,
                    current_memory=state.memory, limit=max(limit * 4, 24), max_chars=max_chars * 4)
        for record in archives:
            content = record["content"]
            fact = by_id.get(content.get("fact_id")) if record["kind"] == "fact" and isinstance(content, dict) else None
            if fact:  # SQLite owns status, including deleted/conflict; never resurrect via archive.
                continue
            if record["kind"] == "fact" and isinstance(content, dict) and digest(
                    [content.get("text"), sorted(content.get("evidence_ids", []))]) in fact_signatures:
                continue
            if record.get("superseded") and not history:
                continue
            if record.get("turn_id") in event_turns and record["kind"] == "message":
                continue  # Original turn already has an episodic representation (even if filtered out).
            archive_papers = []
            if isinstance(content, dict):
                archive_papers = decode_papers(content.get("paper_ids", content.get("paper_id")))
                answer = content.get("content", {})
                if isinstance(answer, dict):
                    archive_papers += [r["paper_id"] for c in answer.get("claims", []) for r in c.get("citations", [])]
            if scope and not scope.intersection(archive_papers):
                continue
            text = content.get("text", json.dumps(content, ensure_ascii=False)) if isinstance(content, dict) else str(content)
            add({**record, "paper_ids": sorted(set(archive_papers)), "created_at": 0}, "archive", lexical(text),
                "query_coverage", "superseded" if record.get("superseded") else "historical",
                "archive:" + digest([state.session_id, record["kind"], content]))
        return self.rank_memories(records, limit, max_chars)

    def compact_working_memory(self, state, strategy, reasoner):
        from .pipeline import compact_state
        if strategy == "semantic":
            from .compaction import semantic_compact_state
            try:
                return semantic_compact_state(state, self.root/"archives", reasoner)
            except Exception as exc:
                self.events.append({"operation": "semantic_compaction", "degraded": True,
                                    "error_type": type(exc).__name__, "recovery": "deterministic_compaction"})
        return compact_state(state, self.root/"archives")

    def commit(self, state, question, answer, evidence, turn_id, base_version, compact, strategy, reasoner):
        candidates, episode = self.extract_memories(question, answer, evidence, turn_id)
        if episode:
            try:
                episode["similar_episode_ids"] = [e["episode_id"] for e in self.episodic.search(state.session_id, question, limit=3)]
            except Exception as exc:
                self._failure("chroma_similarity", exc)
                episode["similar_episode_ids"] = []
        with self.facts.transaction() as db:
            previous = db.execute("SELECT version FROM sessions WHERE session_id=?", (state.session_id,)).fetchone()
            if previous and previous[0] != base_version:
                raise RuntimeError("Concurrent session update; reload and retry with the same turn_id")
            if previous is None:
                legacy = [{"key": key, "text": f["text"], "source_type": "conversation",
                           "status": "conflict", "reason": "legacy_unverified", "evidence_ids": f.get("evidence_ids", [])}
                          for key, f in state.memory.facts.items()]
                self.update_fact_memory(db, state.session_id, "legacy_import", legacy)
            self.events.extend(self.update_fact_memory(db, state.session_id, turn_id, candidates))
            self.events.append(self.write_episode(db, state.session_id, episode))
            active = db.execute("SELECT * FROM facts WHERE session_id=? AND status='active' ORDER BY created_at",
                                (state.session_id,)).fetchall()
            state.memory.facts = {f["canonical_key"]: {"text": f["text"], "evidence_ids": json.loads(f["evidence_ids"]),
                "fact_id": f["fact_id"], "turn_id": f["turn_id"], "source_type": f["source_type"]} for f in active}
            compression = self.compact_working_memory(state, strategy, reasoner) if compact else {"compressed": False}
            db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
                       (state.session_id, state.turn_count, json.dumps(asdict(state), ensure_ascii=False)))
        # The durable transaction is the commit point. Projection failure cannot undo it.
        try:
            self.repair_projections(state)
        except Exception as exc:
            self._failure("projection_recovery", exc, committed=True)
        return compression
