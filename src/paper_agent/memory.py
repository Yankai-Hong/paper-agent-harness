"""Session memory, deliberately separate from s09 persistent preferences.

Archive/preview mechanism inspired by learn-claude-code s08 (MIT).
Summarization is injected: no hidden LLM or character-based token savings claims.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

from .storage import digest, safe_id, write_json


@dataclass
class SessionMemory:
    session_id: str
    entities: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    unresolved: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def update(self, entities, facts, unresolved, valid_evidence):
        validated = {}
        for fact in facts:
            ids = fact.get("evidence_ids", [])
            if not ids or not set(ids) <= set(valid_evidence):
                raise ValueError("Memory fact must reference available evidence")
            validated[fact["key"]] = {"text": fact["text"], "evidence_ids": ids}
        for key, value in entities.items():
            self.entities.pop(key, None)
            self.entities[key] = value
        for key, value in validated.items():
            self.facts.pop(key, None)
            self.facts[key] = value
        self.unresolved = list(unresolved)

    def save(self, root):
        write_json(Path(root)/f"{safe_id(self.session_id)}.json", asdict(self))

    @classmethod
    def load(cls, root, session_id):
        snapshot = Path(root)/f"{safe_id(session_id)}.state.json"
        if snapshot.exists():
            data = json.loads(snapshot.read_text(encoding="utf-8"))["memory"]
            if data["session_id"] != session_id:
                raise ValueError("Session identity mismatch")
            return cls(**data)
        path = Path(root)/f"{safe_id(session_id)}.json"
        if not path.exists():
            return cls(session_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["session_id"] != session_id:
            raise ValueError("Session identity mismatch")
        return cls(**data)


class ContextArchive:
    def __init__(self, root):
        self.root = Path(root)

    def store(self, value):
        identifier = digest(value)
        write_json(self.root/f"{identifier}.json", value)
        return identifier

    def read(self, identifier):
        safe_id(identifier)
        value = json.loads((self.root/f"{identifier}.json").read_text(encoding="utf-8"))
        if digest(value) != identifier:
            raise ValueError("Archive hash mismatch")
        return value

    def search(self, session_id, query, current_memory=None, paper_ids=None, entity=None,
               fact_key=None, before_turn=None, limit=6, max_chars=5000, committed_state=None):
        """Bounded lexical recall of session snapshots; records remain context, never evidence."""
        from .corpus import lexical_tokens
        safe_id(session_id)
        if committed_state is None:
            # Legacy's atomic file is authoritative. Three-layer callers supply SQLite.
            path = self.root.parent / f"{session_id}.state.json"
            committed_state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        allowed = set(committed_state.get("archive_ids", []))
        ledger = committed_state.get("committed_turns", {})
        terms = set(lexical_tokens(query))
        snapshots = []
        for path in self.root.glob("*.json"):
            value = self.read(path.stem)
            if not isinstance(value, dict) or value.get("memory", {}).get("session_id") != session_id:
                continue
            order = value.get("turn_count", 0)
            if path.stem not in allowed:
                if value.get("commit_protocol"):
                    continue
                # Old archives have no manifest. Require every original message to
                # match the committed turn ledger, not merely a reused turn number.
                messages = value.get("messages", [])
                if not messages or order > committed_state.get("turn_count", 0):
                    continue
                def committed_message(message):
                    turn = ledger.get(message.get("turn_id"), {})
                    expected = turn.get("question") if message.get("role") == "user" else turn.get("result", {}).get("answer")
                    return expected is not None and message.get("content") == expected
                if not all(committed_message(m) for m in messages):
                    continue
                # Fact records must also be attributable to those committed claims.
                claims = [c for t in ledger.values() for c in t.get("result", {}).get("answer", {}).get("claims", [])]
                if any(not any(f.get("text") == c.get("text") and
                               set(f.get("evidence_ids", [])) == {r["chunk_id"] for r in c.get("citations", [])}
                               for c in claims) for f in value["memory"].get("facts", {}).values()):
                    continue
            if before_turn is not None and order >= before_turn:
                continue
            snapshots.append((order, path.stem, value))
        snapshots.sort(key=lambda x: (x[0], x[1]))
        latest = {}
        for _, _, value in snapshots:
            latest.update(value["memory"].get("facts", {}))
        if current_memory:
            latest.update(current_memory.facts)
        records = {}
        for order, archive_id, value in snapshots:
            memory = value["memory"]
            entities = memory.get("entities", {})
            for key, fact in memory.get("facts", {}).items():
                record = {"archive_id": archive_id, "turn_id": fact.get("turn_id"),
                          "turn_order": order, "kind": "fact", "key": key, "content": fact,
                          "entities": entities, "evidence_ids": fact.get("evidence_ids", []),
                          "superseded": latest.get(key) != fact}
                records[("fact", key, digest(fact))] = record
            for message in value.get("messages", []):
                claims = message.get("content", {}).get("claims", []) if isinstance(message.get("content"), dict) else []
                superseded = any(c.get("key") in latest and latest[c["key"]].get("text") != c.get("text") for c in claims)
                record = {"archive_id": archive_id, "turn_id": message.get("turn_id"),
                          "turn_order": order, "kind": "message", "content": message,
                          "entities": entities, "evidence_ids": [x["chunk_id"] for c in claims for x in c.get("citations", [])],
                          "superseded": superseded}
                records[("message", digest(message))] = record
        ranked = []
        for record in records.values():
            content = json.dumps(record["content"], ensure_ascii=False)
            score = len(terms & set(lexical_tokens(content)))
            if fact_key and record.get("key") == fact_key: score += 10
            if entity and entity in record["entities"]: score += 5
            if paper_ids and any(p in content for p in paper_ids): score += 5
            if score:
                record["score"] = score
                ranked.append(record)
        ranked.sort(key=lambda r: (r["superseded"], -r["score"], -r["turn_order"], r["archive_id"]))
        selected, used = [], 0
        for record in ranked:
            size = len(json.dumps(record, ensure_ascii=False))
            if used + size <= max_chars:
                selected.append(record)
                used += size
            if len(selected) >= limit: break
        return selected

    def compress_tool_results(self, messages, max_chars=2000, preserve_last=3):
        import copy
        if max_chars < 1 or preserve_last < 0:
            raise ValueError("Invalid compaction budget")
        output = copy.deepcopy(messages)
        positions = [(mi, bi) for mi, message in enumerate(output)
                     if isinstance(message.get("content"), list)
                     for bi, block in enumerate(message["content"]) if block.get("type") == "tool_result"]
        candidates = positions[:-preserve_last] if preserve_last else positions
        for mi, bi in candidates:
            if not any(m.get("role") == "assistant" for m in output[mi+1:]):
                continue  # Do not shorten results that the model has not consumed.
            block = output[mi]["content"][bi]
            text = block.get("content", "")
            if isinstance(text, str) and len(text) > max_chars:
                ref = self.store(text)
                block["content"] = json.dumps({"archive_id": ref, "preview": text[:max_chars],
                                                "read_tool": "read_archive"}, ensure_ascii=False)
        return output


def validate_citations(claims, evidence):
    """Structural validation only; semantic support needs a separate evaluation."""
    by_id = {e["chunk_id"]: e for e in evidence}
    for claim in claims:
        if not claim.get("text", "").strip() or not claim.get("citations"):
            raise ValueError("Every factual claim needs text and citations")
        for cite in claim["citations"]:
            chunk = by_id.get(cite["chunk_id"])
            if chunk is None or chunk["paper_id"] != cite["paper_id"]:
                raise ValueError("Citation does not match available evidence")
    return True
