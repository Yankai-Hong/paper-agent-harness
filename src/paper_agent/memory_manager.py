"""Three-layer coordination; historical memory is never paper evidence."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import re
import time

from .memory import SessionMemory, ContextArchive, validate_citations
from .long_term import MemoryStore
from .memory_stores import RedisWorkingStore, ChromaEpisodes, paper_ids as decode_papers
from .storage import digest


def extract_declarations(question, answer, turn_id):
    """Rule labels, not semantic inference. Paper answers never become memories."""
    candidates=[]
    match=re.fullmatch(r'(记住|偏好|约束|决定|更新记忆|Remember|Preference|Constraint|Decision|Update memory)[:：]\s*([^=：:]+)[=：:]\s*(.+)',question,re.I|re.S)
    if match:
        marker=match[1].casefold()
        candidates.append({'key':match[2].strip(),'text':match[3].strip(),
            'kind':'constraint' if marker in {'约束','constraint'} else 'decision' if marker in {'决定','decision'} else 'preference',
            'explicit_update':marker in {'更新记忆','update memory'},'source_type':'user_explicit','status':'active'})
    event_type='user_declaration' if match else None
    for kind,pattern in (
        ('research_decision',r'^(研究决策|实现决策|决定|决策|Decision)[:：]|^(我|我们)决定'),
        ('experiment_result',r'^(关键实验结果|实验结果|Experiment result)[:：]'),
        ('method_change',r'^(方法变更|Method change)[:：]'),
        ('follow_up',r'^(后续待办|未解决问题|TODO|Unresolved)[:：]'),
        ('research_note',r'^(复用笔记|研究笔记|Research note)[:：]'),
        ('user_constraint',r'^(研究约束|Research constraint)[:：]')):
        if re.search(pattern,question.strip(),re.I): event_type=kind; break
    episode=None
    if event_type:
        refs=[r for c in answer.get('claims',[]) for r in c.get('citations',[])]
        episode={'turn_id':turn_id,'content':json.dumps({'question':question,'answer':answer},ensure_ascii=False),
                 'event_type':event_type,'created_at':time.time(),'source_type':'conversation',
                 'paper_ids':sorted({r['paper_id'] for r in refs}),'evidence_ids':sorted({r['chunk_id'] for r in refs})}
    return candidates,episode


class MemoryManager:
    def __init__(self, root, working, episodic, fallback=True, user_id="local", project_id=None, share_project=False):
        self.identity = (digest([user_id, str(Path(root).resolve())]) if project_id is None else
                         digest([user_id, project_id, str(Path(root).resolve())]))
        self.root = Path(root) if user_id == 'local' and project_id is None else Path(root)/'scopes'/self.identity
        self.store = MemoryStore(self.root / 'memory.sqlite3')
        self.share_project = share_project
        # Default legacy directory is local-only; records remain unpromoted without ownership review.
        from .migration import migrate, migrate_file_snapshots
        self.migration = migrate(self.root/'facts.sqlite3', self.store)
        # A separately migrated target remembers its original source path; do
        # not attempt to treat the target directory itself as a second source.
        with self.store.transaction() as db:
            file_audit=db.execute('SELECT audit FROM migrations WHERE version=2').fetchone()
        self.file_migration = json.loads(file_audit[0]) if file_audit else migrate_file_snapshots(self.root, self.store)
        if isinstance(getattr(working,'namespace',None),str):
            previous=getattr(working,'_paper_agent_identity',None)
            if previous and previous != self.identity:
                raise ValueError('Redis projection already bound to another memory namespace')
            if not previous:
                working.namespace += ':' + self.identity
                working._paper_agent_identity = self.identity
        if isinstance(episodic, ChromaEpisodes):
            previous=getattr(episodic,'_paper_agent_identity',None)
            if previous and previous != self.identity:
                raise ValueError('Chroma projection already bound to another memory namespace')
            if not previous:
                if episodic._collection is not None:
                    raise ValueError('Chroma projection must be bound before initialization')
                episodic.name += '_' + self.identity[:16]
                episodic._paper_agent_identity = self.identity
        self.working, self.episodic, self.fallback = working, episodic, fallback
        self.events = []

    @classmethod
    def local(cls, root):
        return cls(root,None,None,user_id=os.getenv('MEMORY_LOCAL_USER','local'),
                   project_id=os.getenv('MEMORY_PROJECT') or None,
                   share_project=os.getenv('MEMORY_SHARE_PROJECT','false').lower()=='true')

    @classmethod
    def from_env(cls, root, encoder):
        return cls(root, RedisWorkingStore(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                   namespace=os.getenv("MEMORY_REDIS_NAMESPACE", "paper-agent:working:v1"),
                   ttl=int(os.getenv("MEMORY_WORKING_TTL", "86400"))),
                   ChromaEpisodes(os.getenv("MEMORY_CHROMA_PATH", str(Path(root)/"chroma")), encoder,
                                  collection=os.getenv("MEMORY_CHROMA_COLLECTION", "paper_agent_episodes_v1"),
                                  min_similarity=float(os.getenv("MEMORY_MIN_SIMILARITY", ".35"))),
                   fallback=os.getenv("MEMORY_ALLOW_FALLBACK", "true").lower() == "true",
                   user_id=os.getenv('MEMORY_LOCAL_USER','local'), project_id=os.getenv('MEMORY_PROJECT') or None,
                   share_project=os.getenv('MEMORY_SHARE_PROJECT','false').lower()=='true')

    def _failure(self, operation, exc, committed=False):
        # Never expose connection URLs/passwords through an exception message.
        self.events.append({"operation": operation, "degraded": True, "error_type": type(exc).__name__,
                            "recovery": "durable_sqlite_and_file", "committed": committed})
        if not self.fallback and not committed:
            raise RuntimeError(f"Memory {operation} failed ({type(exc).__name__})") from None

    def load_working_memory(self, state, bounded=True):
        if state.memory_namespace and state.memory_namespace != self.identity:
            raise ValueError('Working state belongs to another local user/project')
        state.memory_namespace = self.identity
        durable = self.store.snapshot(state.session_id)
        cached = None
        try:
            cached = self.working.load(state.session_id) if self.working else None
            if cached and not durable:
                self.events.append({"operation": "ignore_orphan_cache"})
        except Exception as exc:
            self._failure("redis_read", exc)
        if durable is None:
            # File projections are never authority. Old independent files need an
            # explicit ownership-confirmed migration before their contents are used.
            if state.turn_count and state.session_id in self.file_migration.get('sessions_found',[]):
                raise ValueError('Legacy file session requires ownership-confirmed migration')
        else:
            # Accept only a cache projection matching the durable committed version/content.
            durable_view = {**durable, "committed_turns": {}}
            if cached == durable_view:
                durable = {**cached, "committed_turns": durable["committed_turns"]}
                self.events.append({"operation": "redis_cache_hit"})
            durable["memory"] = SessionMemory.from_snapshot(durable["memory"])
            state.__dict__.update(durable)
        state.memory_namespace = self.identity
        self.repair_projections(state)
        return state

    def update_working_memory(self, state, committed=False):
        if self.working is None:
            return
        try:
            # Keep the unbounded idempotency ledger in SQLite/file, not Redis.
            snapshot = asdict(state)
            snapshot["committed_turns"] = {}
            snapshot["messages"] = snapshot["messages"][-8:]
            self.working.save(snapshot)
        except Exception as exc:
            self._failure("redis_write", exc, committed)

    def repair_projections(self, state):
        self.check_episode_index(state.session_id)
        # Repeating an upsert after a crash is safe, including crash after index write.
        for episode in (self.store.episodes(state.session_id, pending=True, limit=32) if self.episodic else []):
            try:
                self.episodic.write(episode)
                with self.store.transaction() as db:
                    db.execute("UPDATE episodes SET indexed=1 WHERE episode_id=?", (episode["episode_id"],))
            except Exception as exc:
                self._failure("chroma_outbox", exc, committed=True)
                break
        if self.store.snapshot(state.session_id) is not None:
            try:
                state.save(self.root)
            except Exception as exc:
                self._failure("file_projection", exc, committed=True)
            self.update_working_memory(state, committed=True)

    def check_episode_index(self, session_id):
        if self.episodic is None:
            return
        batch = self.store.audit_batch(session_id)
        if not batch:
            return
        try:
            indexed = self.episodic.fingerprints(session_id, [e["episode_id"] for e in batch])
            missing = [e["episode_id"] for e in batch if indexed.get(e["episode_id"]) != digest(e)]
        except Exception as exc:
            # Retain checked records for retry; do not advance the audit cursor on failure.
            with self.store.transaction() as db:
                db.executemany("UPDATE episodes SET indexed=0 WHERE episode_id=? AND session_id=?",
                               [(e["episode_id"], session_id) for e in batch])
            self._failure("chroma_consistency", exc, committed=True)
            return
        with self.store.transaction() as db:
            db.executemany("UPDATE episodes SET indexed=0 WHERE episode_id=? AND session_id=?",
                           [(identifier, session_id) for identifier in missing])
            db.execute("INSERT OR REPLACE INTO episode_audit_cursor VALUES (?,?)", (session_id, batch[-1]["episode_id"]))
        if missing:
            self.events.append({"operation": "chroma_repair_queued", "episode_ids": missing})

    def extract_memories(self, question, answer, evidence, turn_id):
        if answer.get('decision') == 'answer':
            validate_citations(answer['claims'], evidence)
        candidates, episode = extract_declarations(question, answer, turn_id)
        if self.share_project:
            for candidate in candidates:
                candidate['scope'] = 'project'
        return candidates, episode

    def write_episode(self, db, session_id, episode):
        if episode is None:
            return {"action": "skip_low_value"}
        episode = {**episode, "session_id": session_id, "share_project": self.share_project}
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
        if state.memory_namespace and state.memory_namespace != self.identity:
            raise ValueError('Memory identity mismatch')
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
            record = {**record, "source_session_id": record.get('session_id',state.session_id),
                      "session_id": state.session_id, "record_id": identity,
                      "retrieval_source": source, "source_type": record.get("source_type", "conversation"),
                      "status": status, "relevance_score": score, "score_kind": score_kind,
                      "relevance_tier": 2 if score >= (.6 if score_kind == "query_coverage" else .65) else 1,
                      "historical_context_not_evidence": True, "is_authoritative": False,
                      "is_current_memory": source == "sqlite" and status == "active",
                      "requires_resolution": status == "conflict"}
            if status == "conflict":
                record["kind"] = "conflict"
            records.append(record)
        memories = self.store.memories(state.session_id, shared=self.share_project)
        for memory in memories:
            if memory['status'] not in ({'active','conflict','superseded'} if history else {'active','conflict'}):
                continue
            score = lexical(memory['attribute']+' '+memory['text'])
            # Applicable explicit preferences must not be excluded by paper scope.
            if memory['status']=='active' and memory['kind'] in {'preference','constraint'}:
                score=max(.6,score)
            add({**memory, 'kind':'long_term', 'content':memory['text']}, 'sqlite',score,
                'query_coverage',memory['status'],'long_term:'+memory['memory_id'])
        episodes = {e["episode_id"]: e for e in self.store.episodes(state.session_id)}
        if self.share_project:
            with self.store.transaction() as db:
                shared_events=[json.loads(r[0]) for r in db.execute('SELECT payload FROM episodes WHERE session_id<>?',(state.session_id,))]
            for e in shared_events:
                if e.get('share_project'):
                    episodes[e['episode_id']]=e
        self.check_episode_index(state.session_id)
        try:
            hits = self.episodic.search(state.session_id, query, paper_ids, event_type, max(limit * 4, 24)) if self.episodic else [
                {'episode_id':i,'relevance_score':lexical(e['content']),'score_kind':'query_coverage'} for i,e in episodes.items()]
        except Exception as exc:
            self._failure("chroma_read", exc)
            hits = [{"episode_id": i, "relevance_score": lexical(e["content"]),
                     "score_kind": "query_coverage"} for i, e in episodes.items()]
        if self.share_project:
            hits += [{'episode_id':i,'relevance_score':lexical(e['content']),'score_kind':'query_coverage'}
                     for i,e in episodes.items() if e['session_id'] != state.session_id]
        # Missing/corrupt projections still permit clearly labelled lexical recall.
        seen_hits={h['episode_id'] for h in hits}
        hits += [{'episode_id':e['episode_id'],'relevance_score':lexical(e['content']),'score_kind':'query_coverage'}
                 for e in self.store.episodes(state.session_id,pending=True) if e['episode_id'] not in seen_hits]
        for hit in hits:
            episode = episodes.get(hit["episode_id"])
            if not episode or episode.get('status')=='archived' or (scope and episode['paper_ids'] and not scope.intersection(decode_papers(episode["paper_ids"]))):
                continue
            if event_type and episode["event_type"] != event_type:
                continue
            # A historical declaration should not resurrect an archived or
            # superseded preference during ordinary recall. Its event remains
            # available for an explicit history query.
            related = [m for m in memories if m['session_id'] == episode['session_id']
                       and m['turn_id'] == episode['turn_id']]
            if related and all(m['status'] in {'superseded','archived'} for m in related) and not history:
                continue
            if hit["score_kind"] != "query_coverage" and hit.get("payload_hash") != digest(episode):
                with self.store.transaction() as db:
                    db.execute("UPDATE episodes SET indexed=0 WHERE episode_id=?", (hit["episode_id"],))
                continue  # Stale index version is never served as a current candidate.
            add({**episode, "kind": "episode", "distance": hit.get("distance"), "metric": hit.get("metric")},
                "chroma", hit["relevance_score"], hit["score_kind"], "historical",
                "event:" + episode['session_id'] + ":" + episode["turn_id"])
        event_turns = {e["turn_id"] for e in episodes.values()}
        archives = ContextArchive(self.root/"archives").search(state.session_id, query,
                    current_memory=state.memory, limit=max(limit * 4, 24), max_chars=max_chars * 4,
                    committed_state=self.store.snapshot(state.session_id) or {})
        for record in archives:
            content = record["content"]
            if record['kind'] != 'message':
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
        if episode and self.episodic:
            try:
                episode["similar_episode_ids"] = [e["episode_id"] for e in self.episodic.search(state.session_id, question, limit=3)]
            except Exception as exc:
                self._failure("chroma_similarity", exc)
                episode["similar_episode_ids"] = []
        with self.store.transaction() as db:
            previous = db.execute("SELECT version FROM sessions WHERE session_id=?", (state.session_id,)).fetchone()
            if previous and previous[0] != base_version:
                raise RuntimeError("Concurrent session update; reload and retry with the same turn_id")
            self.events.extend(self.store.put(db,state.session_id,turn_id,c,shared=self.share_project) for c in candidates)
            self.events.append(self.write_episode(db,state.session_id,episode))
            compression = self.compact_working_memory(state, strategy, reasoner) if compact else {"compressed": False}
            from .budgets import current_budget
            if current_budget.get():
                current_budget.get().remaining()
            db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
                       (state.session_id, state.turn_count, json.dumps(asdict(state), ensure_ascii=False)))
        # The durable transaction is the commit point. Projection failure cannot undo it.
        try:
            self.repair_projections(state)
        except Exception as exc:
            self._failure("projection_recovery", exc, committed=True)
        return compression
