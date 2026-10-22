import json
from types import SimpleNamespace as NS

import pytest

from paper_agent.corpus import Chunker
from paper_agent.evaluation import retrieval_metrics
from paper_agent.memory import SessionMemory, ContextArchive, validate_citations
from paper_agent.retrieval import rrf, build_index, Retriever
from paper_agent.runtime import tool_loop
from paper_agent.storage import write_jsonl
from paper_agent.pipeline import PaperPipeline, PaperAgentState


def paper(pid="p", text="alpha beta gamma delta epsilon zeta eta theta"):
    return {"paper_id": pid, "title": pid, "paragraphs": [
        {"paragraph_id": "a", "section": "Methods", "text": text}]}


def test_chunks_preserve_offsets_and_do_not_cross_sections():
    source = paper()
    source["paragraphs"].append({"paragraph_id": "b", "section": "Results", "text": "one two"})
    chunks = Chunker(size=4, overlap=1).split(source)
    for chunk in chunks:
        for span in chunk["spans"]:
            paragraph = next(p for p in source["paragraphs"] if p["paragraph_id"] == span["paragraph_id"])
            assert paragraph["text"][span["start"]:span["end"]] == span["text"]
        assert len({s["section"] for s in chunk["spans"]}) == 1
    assert chunks == Chunker(4, 1).split(source)
    assert "delta" in chunks[0]["text"] and "delta" in chunks[1]["text"]
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_rrf_uses_ranks_and_deduplicates():
    ranking = rrf([["a", "b"], ["b", "c"]])
    assert ranking[0][0] == "b"
    assert dict(ranking)["b"] == pytest.approx(1/62 + 1/61)
    assert dict(rrf([["a", "a"]]))["a"] == pytest.approx(1/61)


def test_recall_dedup_missing_prediction_and_alternatives():
    chunk = Chunker().split(paper())[0]
    queries = [{"query_id": "q1", "gold_units": ["p::a", "p::b"]},
               {"query_id": "q2", "gold_units": ["p::a"]}]
    out = retrieval_metrics(queries, [{"query_id": "q1", "hits": [chunk, chunk]}])
    assert out["Recall@5"] == .25
    assert out["Hit@5"] == .5
    assert out["missing"] == 1
    queries[0]["gold_alternatives"] = [["p::a"], ["p::b"]]
    assert retrieval_metrics(queries, [{"query_id": "q1", "hits": [chunk]}])["Recall@5"] == .5


def test_memory_validates_before_mutation_and_isolates_sessions(tmp_path):
    memory = SessionMemory("first")
    memory.update({"it": "paper-a"}, [{"key": "k", "text": "fact", "evidence_ids": ["c1"]}], [], ["c1"])
    memory.update({"it": "paper-b"}, [], [], [])
    assert memory.entities["it"] == "paper-b"
    with pytest.raises(ValueError):
        memory.update({"it": "bad"}, [{"key": "k", "text": "wrong", "evidence_ids": ["fake"]}], [], [])
    assert memory.entities["it"] == "paper-b"
    memory.save(tmp_path)
    assert SessionMemory.load(tmp_path, "first").facts == memory.facts
    assert SessionMemory.load(tmp_path, "second").facts == {}
    with pytest.raises(ValueError):
        SessionMemory.load(tmp_path, "../escape")


def test_archive_roundtrip_protects_pairs(tmp_path):
    messages = []
    for i in range(4):
        messages.extend([{"role": "assistant", "content": [{"type": "tool_use", "id": str(i)}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": str(i), "content": "x"*100}]}])
    archive = ContextArchive(tmp_path)
    result = archive.compress_tool_results(messages, max_chars=20, preserve_last=1)
    assert result[-1] == messages[-1]
    for i in range(3):
        item = result[i*2+1]["content"][0]
        assert item["tool_use_id"] == result[i*2]["content"][0]["id"]
        assert archive.read(json.loads(item["content"])["archive_id"]) == "x"*100
    assert messages[1]["content"][0]["content"] == "x"*100


def test_citations_reject_wrong_paper_and_missing_ids():
    evidence = Chunker().split(paper())
    claim = {"text": "Fact", "citations": [{"paper_id": "wrong", "chunk_id": evidence[0]["chunk_id"]}]}
    with pytest.raises(ValueError):
        validate_citations([claim], evidence)


def test_tool_loop_fills_all_ids_and_errors():
    responses = iter([
        NS(content=[{"type": "tool_use", "id": "1", "name": "read", "input": {"x": 2}},
                    {"type": "tool_use", "id": "2", "name": "unknown", "input": {}}],
           usage=NS(input_tokens=10, output_tokens=3)),
        NS(content=[{"type": "text", "text": "done"}], usage=NS(input_tokens=20, output_tokens=4))])
    client = NS(messages=NS(create=lambda **kwargs: next(responses)))
    history = [{"role": "user", "content": "question"}]
    result = tool_loop(client, "scripted", history, "system",
        [{"name": "read", "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}],
        {"read": lambda x: x*2})
    results = history[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["1", "2"]
    assert results[1]["is_error"]
    assert result.usage.input_tokens == 30 and result.text == "done"


@pytest.fixture
def retriever(tmp_path):
    chunks = [c for p in [paper("a", "retrieval indexing sparse matching"),
                          paper("b", "translation language grammar evaluation"),
                          paper("c", "graphs edges vertices neural networks")] for c in Chunker().split(p)]
    path = tmp_path/"chunks.jsonl"
    write_jsonl(path, chunks)
    build_index(path, tmp_path/"index")
    return Retriever(tmp_path/"index")


def test_bm25_filter_and_no_silent_dense_fallback(retriever):
    assert retriever.search("retrieval", mode="bm25")["hits"][0]["paper_id"] == "a"
    result = retriever.search("retrieval", mode="bm25", paper_ids=["b"])
    assert {h["paper_id"] for h in result["hits"]} == {"b"}
    assert retriever.search("retrieval", mode="bm25", paper_ids=[])["hits"] == []
    with pytest.raises(ValueError):
        retriever.search("retrieval", mode="dense")


def test_pipeline_limits_retry_and_records_skip(tmp_path, retriever):
    class Reasoner:
        def structured(self, stage, payload):
            if stage == "nlu":
                return {"intent": "fact", "standalone_query": "retrieval", "resolved_entities": {}}
            if stage == "reflect":
                return {"decision": "retrieve_again", "next_query": "indexing"}
            raise AssertionError("Should refuse without calling generator")
    pipeline = PaperPipeline(retriever, Reasoner(), tmp_path/"traces", tmp_path/"memory", mode="bm25")
    state = PaperAgentState("s1")
    result = pipeline.run("query", state)
    assert result["answer"]["decision"] == "refuse"
    trace = json.loads((tmp_path/"traces"/f"{result['trace_id']}.json").read_text())
    assert sum(s["stage"] == "Retrieve" for s in trace["stages"]) == 2
    assert state.memory.facts == {}


def test_pipeline_failure_persists_error_and_skipped(tmp_path, retriever):
    class BadReasoner:
        def structured(self, stage, payload):
            return {"intent": "invalid"}
    pipeline = PaperPipeline(retriever, BadReasoner(), tmp_path/"trace", tmp_path/"memory", mode="bm25")
    with pytest.raises(ValueError):
        pipeline.run("query", PaperAgentState("s"))
    trace = json.loads(next((tmp_path/"trace").glob("*.json")).read_text())
    assert trace["status"] == "error"
    assert next(s for s in trace["stages"] if s["stage"] == "NLU")["status"] == "error"
    assert next(s for s in trace["stages"] if s["stage"] == "Generate")["status"] == "skipped"


def test_pipeline_success_carries_real_chunk_ids_into_memory(tmp_path, retriever):
    class ScriptedReasoner:
        def structured(self, stage, payload):
            if stage == "nlu":
                return {"intent": "fact", "standalone_query": "retrieval", "resolved_entities": {"it": "a"}}
            if stage == "reflect":
                return {"decision": "answer"}
            evidence = payload["evidence"][0]
            return {"claims": [{"text": evidence["text"], "citations": [
                {"paper_id": evidence["paper_id"], "chunk_id": evidence["chunk_id"]}]}]}
    pipeline = PaperPipeline(retriever, ScriptedReasoner(), tmp_path/"trace", tmp_path/"memory", mode="bm25")
    state = PaperAgentState("success")
    result = pipeline.run("retrieval", state)
    assert result["answer"]["decision"] == "answer"
    restored = SessionMemory.load(tmp_path/"memory", "success")
    assert restored.entities == {"it": "a"}
    assert all(set(f["evidence_ids"]) <= set(retriever.by_id) for f in restored.facts.values())
    trace = json.loads((tmp_path/"trace"/f"{result['trace_id']}.json").read_text())
    assert all(event["status"] == "ok" for event in trace["stages"])


def test_guardrails_rejects_control_characters(tmp_path, retriever):
    class Reasoner:
        def structured(self, stage, payload):
            return {"intent": "fact", "standalone_query": "x", "resolved_entities": {}}
    pipeline = PaperPipeline(retriever, Reasoner(), tmp_path/"trace", tmp_path/"memory", mode="bm25")
    with pytest.raises(ValueError, match="control"):
        pipeline.run("bad\x00input", PaperAgentState("g1"))
    trace = json.loads(next((tmp_path/"trace").glob("*.json")).read_text())
    assert next(s for s in trace["stages"] if s["stage"] == "Guardrails")["status"] == "error"


def test_citation_validation_failure_is_controlled_refusal(tmp_path, retriever):
    class BadCitationReasoner:
        def structured(self, stage, payload):
            if stage == "nlu":
                return {"intent": "fact", "standalone_query": "retrieval", "resolved_entities": {}}
            if stage == "reflect":
                return {"decision": "answer", "reason": "ok"}
            return {"claims": [{"key": "k", "text": "fact", "citations": [
                {"paper_id": "p", "chunk_id": "nonexistent"}]}]}
    pipeline = PaperPipeline(retriever, BadCitationReasoner(), tmp_path/"trace", tmp_path/"memory", mode="bm25")
    result = pipeline.run("query", PaperAgentState("c1"))
    assert result["answer"]["decision"] == "refuse"
    assert result["answer"]["reason_code"] == "citation_validation_failed"
    trace = json.loads((tmp_path/"trace"/f"{result['trace_id']}.json").read_text())
    assert trace["status"] == "completed"


def test_semantic_compaction_falls_back_to_bounded(tmp_path, retriever):
    class FailingSummaryReasoner:
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
    pipeline = PaperPipeline(retriever, FailingSummaryReasoner(), tmp_path/"trace", tmp_path/"memory",
                             mode="bm25", memory_strategy="semantic")
    state = PaperAgentState("s1")
    for i in range(5):
        state.messages.extend([{"role": "user", "content": f"q{i}"},
                               {"role": "assistant", "content": {"decision": "answer", "claims": []}}])
    result = pipeline.run("query", state)
    assert result["answer"]["decision"] == "answer"
    trace = json.loads((tmp_path/"trace"/f"{result['trace_id']}.json").read_text())
    mem_stage = next(s for s in trace["stages"] if s["stage"] == "Memory")
    assert mem_stage["compaction_fallback"]["from"] == "semantic"
    assert mem_stage["compaction_fallback"]["to"] == "bounded"
