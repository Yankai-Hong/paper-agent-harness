"""Standalone verification of Phase C pipeline fixes (no pytest tmp_path)."""
import json
import shutil
import sys
from pathlib import Path

from paper_agent.corpus import Chunker
from paper_agent.memory import SessionMemory
from paper_agent.pipeline import PaperPipeline, PaperAgentState
from paper_agent.retrieval import build_index, Retriever
from paper_agent.storage import write_jsonl

BASE = Path(__file__).resolve().parents[1] / "verify-c-fixes"
shutil.rmtree(BASE, ignore_errors=True)
BASE.mkdir(parents=True, exist_ok=True)


def make_retriever(root):
    paper = {"paper_id": "p", "title": "Example", "paragraphs": [
        {"paragraph_id": "a", "section": "Methods", "text": "retrieval sparse matching algorithm"}]}
    chunks = Chunker().split(paper)
    write_jsonl(root / "chunks.jsonl", chunks)
    build_index(root / "chunks.jsonl", root / "index")
    return Retriever(root / "index"), chunks


def check_guardrails(retriever):
    class Reasoner:
        calls = []
        def structured(self, stage, payload):
            return {"intent": "fact", "standalone_query": "x", "resolved_entities": {}}
    p = PaperPipeline(retriever, Reasoner(), BASE / "g-trace", BASE / "g-mem", mode="bm25")
    try:
        p.run("bad\x00input", PaperAgentState("g1"))
        return "FAIL: control char not rejected"
    except ValueError as exc:
        return "ok" if "control" in str(exc) else f"FAIL: {exc}"


def check_citation_refuse(retriever):
    class BadCitationReasoner:
        calls = []
        def structured(self, stage, payload):
            if stage == "nlu":
                return {"intent": "fact", "standalone_query": "retrieval", "resolved_entities": {}}
            if stage == "reflect":
                return {"decision": "answer", "reason": "ok"}
            return {"claims": [{"key": "k", "text": "fact", "citations": [
                {"paper_id": "p", "chunk_id": "nonexistent"}]}]}
    p = PaperPipeline(retriever, BadCitationReasoner(), BASE / "c-trace", BASE / "c-mem", mode="bm25")
    result = p.run("query", PaperAgentState("c1"))
    if result["answer"]["decision"] != "refuse":
        return f"FAIL: {result['answer']}"
    if result["answer"].get("reason_code") != "citation_validation_failed":
        return f"FAIL: missing reason_code {result['answer']}"
    trace = json.loads((BASE / "c-trace" / f"{result['trace_id']}.json").read_text())
    if trace["status"] != "completed":
        return f"FAIL: trace status {trace['status']}"
    return "ok"


def check_semantic_fallback(retriever):
    class FailingSummaryReasoner:
        calls = []
        def structured(self, stage, payload):
            if stage == "nlu":
                return {"intent": "fact", "standalone_query": "retrieval", "resolved_entities": {}}
            if stage == "reflect":
                return {"decision": "answer", "reason": "ok"}
            if stage == "generate":
                ev = payload["evidence"][0]
                return {"claims": [{"key": "k", "text": ev["text"], "citations": [
                    {"paper_id": ev["paper_id"], "chunk_id": ev["chunk_id"]}]}]}
            if stage == "memory_summary":
                raise RuntimeError("simulated summary failure")
            raise AssertionError(stage)
    p = PaperPipeline(retriever, FailingSummaryReasoner(), BASE / "s-trace", BASE / "s-mem",
                      mode="bm25", memory_strategy="semantic")
    state = PaperAgentState("s1")
    # Force compaction to trigger by exceeding message cap.
    for i in range(5):
        state.messages.extend([{"role": "user", "content": f"q{i}"},
                               {"role": "assistant", "content": {"decision": "answer", "claims": []}}])
    result = p.run("query", state)
    trace = json.loads((BASE / "s-trace" / f"{result['trace_id']}.json").read_text())
    mem = next(s for s in trace["stages"] if s["stage"] == "Memory")
    fallback = mem.get("output", {}).get("compaction_fallback") if isinstance(mem.get("output"), dict) else None
    # compaction info is on the stage event, stored as output only for memory state;
    # the fallback marker lives on the Memory stage event itself.
    mem_stage = next(s for s in trace["stages"] if s["stage"] == "Memory")
    if "compaction_fallback" not in mem_stage:
        return f"FAIL: no fallback marker in {mem_stage.keys()}"
    if result["answer"]["decision"] != "answer":
        return f"FAIL: answer lost during fallback {result['answer']}"
    return "ok"


if __name__ == "__main__":
    retriever, chunks = make_retriever(BASE)
    results = {
        "guardrails_control_chars": check_guardrails(retriever),
        "citation_invalid_refuses": check_citation_refuse(retriever),
        "semantic_compaction_fallback": check_semantic_fallback(retriever),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    ok = all(v == "ok" for v in results.values())
    print("VERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
