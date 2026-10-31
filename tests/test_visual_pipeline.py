import base64
import hashlib
import json
from pathlib import Path

import pytest

from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.pipeline import PaperPipeline, PaperAgentState
from paper_agent.budgets import TurnBudget, current_budget, BudgetExceeded


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/a6kAAAAASUVORK5CYII=")


def fixture_image(root):
    (root / "figure.png").write_bytes(PNG)
    return {"paper_id":"p", "chunk_id":"visual-1", "text":"Figure 1", "evidence_type":"figure",
            "page":2, "bbox":[1,2,9,10], "page_size":[100,100], "coordinate_system":"top-left-points",
            "page_rotation":0, "figure_number":"Figure 1", "image_path":"figure.png", "image_sha256":hashlib.sha256(PNG).hexdigest()}


def response(value):
    return {"model":"fixture", "choices":[{"finish_reason":"stop", "message":{"content":json.dumps(value)}}],
            "usage":{"prompt_tokens":10,"completion_tokens":3}}


def test_provider_image_blocks_limits_privacy_and_budget(tmp_path):
    ev = fixture_image(tmp_path)
    received = []
    def transport(url, body, headers, timeout):
        received.append((body, headers, timeout))
        return response({"decision":"answer", "reason":"seen"})
    reasoner = CompatibleReasoner(APIConfig("secret", "fixture", max_tokens=1024, timeout=60, max_calls=1, max_retries=0, repair_retries=0), "test-session", transport, image_root=tmp_path)
    token = current_budget.set(TurnBudget(model_requests=1))
    try:
        reasoner.structured("reflect", {"question":"What is shown?", "evidence":[ev]}, images=[ev])
        with pytest.raises(RuntimeError, match="API_MAX_CALLS"):
            reasoner.structured("reflect", {}, images=[ev])
    finally:
        current_budget.reset(token)
    body, headers, timeout = received[0]
    blocks = body["messages"][1]["content"]
    assert [x["type"] for x in blocks] == ["text", "text", "image_url"]
    assert blocks[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert headers["x-opencode-session"] == "test-session" and timeout <= 60
    assert body["max_tokens"] == 1024 and len(reasoner.calls) == 1
    assert reasoner.calls[0]["image_count"] == 1
    assert base64.b64encode(PNG).decode() not in json.dumps(reasoner.calls)
    bad = dict(ev, image_path="../figure.png")
    other = CompatibleReasoner(APIConfig("secret", "fixture"), "s", transport, image_root=tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        other.structured("reflect", {}, images=[bad])
    with pytest.raises(ValueError):
        other.structured("reflect", {}, images=[ev] * 4)
    assert len(received) == 1


class TextRetriever:
    def __init__(self):
        self.item = {"paper_id":"p", "chunk_id":"text-1", "text":"A short paper excerpt.", "title":"Paper"}
        self.by_id = {"text-1": self.item}
        self.metadata = {"corpus_hash":"test"}
        self.reranker = None
    def search(self, *args, **kwargs):
        return {"hits":[self.item]}


class VisualRetriever:
    def __init__(self, root, item):
        self.image_root, self.item = root, item
    def search_text(self, query, **kwargs):
        return {"hits":[self.item], "timing_ms":1}
    def search_caption(self, query, **kwargs):
        return {"hits":[self.item]}


@pytest.mark.parametrize("outcome", ["answer", "reflect_refuse", "generate_refuse", "bad_citation"])
def test_pipeline_reflect_generate_see_images_and_trusted_citation(tmp_path, outcome):
    image_root = tmp_path / "images"
    image_root.mkdir()
    ev = fixture_image(image_root)
    requests = []
    def transport(url, body, headers, timeout):
        requests.append(body)
        stage = len(requests)
        if stage == 1:
            value = {"intent":"fact", "standalone_query":"diagram", "needs_clarification":False, "paper_ids":["p"], "resolved_entities":{}}
        elif stage == 2:
            value = {"decision":"refuse" if outcome == "reflect_refuse" else "answer", "reason":"reviewed"}
        elif outcome == "generate_refuse":
            value = {"decision":"refuse", "reason":"unreadable", "claims":[]}
        else:
            value = {"claims":[{"key":"diagram", "text":"A diagram", "citations":[{"paper_id":"p", "chunk_id":"wrong" if outcome == "bad_citation" else "visual-1"}]}]}
        return response(value)
    reasoner = CompatibleReasoner(APIConfig("secret", "fixture", max_retries=0, repair_retries=0), "s", transport, image_root=image_root)
    pipeline = PaperPipeline(TextRetriever(), reasoner, tmp_path/"traces", tmp_path/"memory", mode="bm25", visual_retriever=VisualRetriever(image_root,ev))
    result = pipeline.run("Describe the diagram", PaperAgentState("s"))
    assert requests[1]["messages"][1]["content"][-1]["type"] == "image_url"
    if outcome == "reflect_refuse":
        assert len(requests) == 2 and result["answer"]["decision"] == "refuse"
    else:
        assert requests[2]["messages"][1]["content"][-1]["type"] == "image_url"
        if outcome == "answer":
            cite = result["answer"]["claims"][0]["citations"][0]
            assert cite["page"] == 2 and cite["bbox"] == ev["bbox"]
            assert cite["image_path"] == str((image_root / "figure.png").resolve())
        else:
            assert result["answer"]["decision"] == "refuse"
    for file in list((tmp_path/"traces").rglob("*.json")) + list((tmp_path/"memory").rglob("*.json")):
        assert base64.b64encode(PNG).decode() not in file.read_text(encoding="utf-8")
    trace_text = Path(result['trace_path']).read_text(encoding='utf-8')
    for private in ('Describe the diagram', 'A short paper excerpt.', 'Figure 1', 'A diagram', 'reviewed', 'unreadable'):
        assert private not in trace_text
    trace = json.loads(trace_text)
    assert trace['privacy'] == 'visual_metadata_only_v1'
    assert all('usage' in call for call in trace['api_calls'])
    rerank = next(stage for stage in trace['stages'] if stage['stage'] == 'Rerank')
    assert 'visual-1' in rerank['output']['evidence_ids']
    assert {'paper_id': 'p', 'chunk_id': 'visual-1'} in rerank['output']['ranked_refs']


def test_conflicting_evidence_key_fails_closed():
    from paper_agent.pipeline import _merge_evidence_entities
    text = {"paper_id":"p", "chunk_id":"same", "text":"text"}
    visual = {"paper_id":"p", "chunk_id":"same", "text":"caption", "evidence_type":"figure"}
    with pytest.raises(ValueError, match="Conflicting evidence"):
        _merge_evidence_entities([text], [visual])
    same = {**visual, "rank":5, "visual_score":0.7}
    assert len(_merge_evidence_entities([visual], [same])) == 1


def test_associated_prose_is_promoted_only_after_index_verification():
    from paper_agent.pipeline import _expand_verified_associations
    sha = 'a' * 64
    text = {'paper_id':'p', 'chunk_id':'t', 'evidence_type':'text', 'page':2,
            'pdf_sha256':sha, 'text':'See Figure 1 for the evidence.'}
    ref = {'paper_id':'p', 'chunk_id':'t', 'page':2, 'text':text['text'],
           'reference_text':'Figure 1', 'reference_span':[4,12],
           'method':'explicit_figure_or_table_reference'}
    visual = {'paper_id':'p', 'chunk_id':'f', 'evidence_type':'figure', 'pdf_sha256':sha,
              'figure_number':'Figure 1', 'associated_text':[ref]}
    evidence, ids = _expand_verified_associations([visual], {'t':text})
    assert ids == ['t'] and evidence[1] == text and 'associated_text' not in evidence[0]
    for invalid in ({**text, 'pdf_sha256':'b'*64}, {**text, 'page':3},
                    {**text, 'paper_id':'other'}, {**text, 'text':'tampered'}):
        clean, ids = _expand_verified_associations([visual], {'t':invalid})
        assert not ids and 'associated_text' not in clean[0]
    clean, ids = _expand_verified_associations([{**visual, 'figure_number':'Figure 2'}], {'t':text})
    assert not ids and 'associated_text' not in clean[0]


def test_caption_entity_mapping_requires_unique_page_hash_and_exact_text():
    from paper_agent.pipeline import _map_exact_caption_entities
    sha = 'a' * 64
    visual = {'paper_id':'p', 'chunk_id':'image', 'evidence_type':'figure', 'page':3,
              'pdf_sha256':sha, 'caption':'Figure 1: A diagram.'}
    caption = {'paper_id':'p', 'chunk_id':'caption', 'evidence_type':'text', 'docling_label':'caption',
               'page':3, 'pdf_sha256':sha, 'text':'Figure 1: A diagram.'}
    text, mapped, ids = _map_exact_caption_entities([caption], [visual])
    assert not text and mapped == [visual] and ids == {('p','caption'):('p','image')}
    for alternatives in ([dict(visual, page=4)], [dict(visual, pdf_sha256='b'*64)],
                         [visual, dict(visual, chunk_id='other')]):
        text, mapped, ids = _map_exact_caption_entities([caption], alternatives)
        assert text == [caption] and not mapped and not ids


def test_caption_and_figure_share_one_citation_entity(tmp_path):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    image = {**fixture_image(image_root), 'caption':'Figure 1', 'pdf_sha256':'a'*64}
    caption = {'paper_id':'p', 'chunk_id':'caption-1', 'text':'Figure 1', 'evidence_type':'text',
               'docling_label':'caption', 'page':2, 'pdf_sha256':'a'*64}
    text = TextRetriever()
    text.search = lambda *args, **kwargs: {'hits':[caption, text.item]}
    class V(VisualRetriever):
        evidence = [image]
    received = []
    def transport(url, body, headers, timeout):
        received.append(body)
        return response({'intent':'fact', 'standalone_query':'diagram', 'needs_clarification':False,
                         'paper_ids':['p'], 'resolved_entities':{}} if len(received) == 1
                        else {'decision':'refuse', 'reason':'insufficient'})
    reasoner = CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0), 's', transport, image_root=image_root)
    pipeline = PaperPipeline(text, reasoner, tmp_path/'traces', tmp_path/'memory', mode='bm25', visual_retriever=V(image_root,image))
    result = pipeline.run('Describe the figure', PaperAgentState('s'))
    trace = json.loads(Path(result['trace_path']).read_text(encoding='utf-8'))
    refs = next(s for s in trace['stages'] if s['stage']=='Rerank')['output']['ranked_refs']
    assert refs.count({'paper_id':'p','chunk_id':'visual-1'}) == 1
    assert not any(ref['chunk_id']=='caption-1' for ref in refs)


def test_mixed_index_scores_visual_caption_only_and_fuses_entity_once(tmp_path, monkeypatch):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    visual = {**fixture_image(image_root), 'evidence_type':'table', 'figure_number':'Table 1',
              'caption':'Table 1: Results', 'text':'Table 1: Results\n| SECRET_CELL |'}
    text = TextRetriever()
    text.search = lambda *args, **kwargs: {'hits':[visual, text.item]}
    class Ranker:
        def score(self, query, texts):
            self.texts = texts
            return [1.0] * len(texts)
    ranker = text.reranker = Ranker()
    import paper_agent.retrieval as retrieval
    original_rrf, rankings_seen = retrieval.rrf, []
    def capture(rankings, k=60):
        rankings_seen.append(rankings)
        return original_rrf(rankings, k)
    monkeypatch.setattr(retrieval, 'rrf', capture)
    class V(VisualRetriever):
        evidence = [visual]
    requests = []
    def transport(url, body, headers, timeout):
        requests.append(body)
        return response({'intent':'fact', 'standalone_query':'results', 'needs_clarification':False,
                         'paper_ids':['p'], 'resolved_entities':{}} if len(requests) == 1
                        else {'decision':'refuse', 'reason':'insufficient'})
    reasoner = CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0), 's', transport, image_root=image_root)
    pipeline = PaperPipeline(text, reasoner, tmp_path/'traces', tmp_path/'memory', mode='rerank', visual_retriever=V(image_root,visual))
    result = pipeline.run('Describe the table', PaperAgentState('s'))
    assert 'Table 1: Results' in ranker.texts
    assert all('SECRET_CELL' not in value for value in ranker.texts)
    trace = json.loads(Path(result['trace_path']).read_text(encoding='utf-8'))
    refs = next(s for s in trace['stages'] if s['stage']=='Rerank')['output']['ranked_refs']
    assert refs.count({'paper_id':'p','chunk_id':'visual-1'}) == 1
    assert rankings_seen[0][0].count(('p','visual-1')) == 1
    assert rankings_seen[0][1].count(('p','visual-1')) == 1
    assert ('p','visual-1') not in rankings_seen[1][0]
    assert rankings_seen[1][1].count(('p','visual-1')) == 1


def test_visual_routes_together_get_one_vote_against_text():
    from paper_agent.pipeline import _fuse_text_visual_candidates
    texts = [{'paper_id':'p','chunk_id':'t1','text':'first text'},
             {'paper_id':'p','chunk_id':'t2','text':'second text'}]
    visuals = [{'paper_id':'p','chunk_id':'v1','text':'first caption','evidence_type':'figure'},
               {'paper_id':'p','chunk_id':'v2','text':'second caption','evidence_type':'figure'}]
    result = _fuse_text_visual_candidates(texts, visuals, visuals)
    assert [(e['paper_id'],e['chunk_id']) for e in result] == [('p','t1'),('p','v1'),('p','t2'),('p','v2')]


@pytest.mark.parametrize('audit_status', ['supported', 'unsupported', 'unknown', 'wrong_key'])
def test_optional_visual_claim_check_views_image_and_fails_closed(tmp_path, audit_status):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    ev = fixture_image(image_root)
    requests = []
    def transport(url, body, headers, timeout):
        requests.append(body)
        if len(requests) == 1:
            return response({'intent':'fact', 'standalone_query':'diagram', 'needs_clarification':False,
                             'paper_ids':['p'], 'resolved_entities':{}})
        if len(requests) == 2:
            return response({'decision':'answer','reason':'evidence seen'})
        if len(requests) == 3:
            return response({'claims':[{'key':'k', 'text':'A diagram',
                                        'citations':[{'paper_id':'p','chunk_id':'visual-1'}]}]})
        return response({'verdicts':[{'claim_key':'different' if audit_status == 'wrong_key' else 'k',
                                      'status':'supported' if audit_status == 'wrong_key' else audit_status}]})
    reasoner = CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0), 's', transport, image_root=image_root)
    pipeline = PaperPipeline(TextRetriever(), reasoner, tmp_path/'traces', tmp_path/'memory', mode='bm25',
                             visual_retriever=VisualRetriever(image_root,ev), visual_claim_check=True)
    result = pipeline.run('Describe the figure', PaperAgentState('s'))
    assert len(requests) == 4 and requests[-1]['messages'][1]['content'][-1]['type'] == 'image_url'
    assert result['answer']['decision'] == ('answer' if audit_status == 'supported' else 'refuse')
    if audit_status != 'supported':
        assert result['answer']['reason_code'] == 'claim_self_check_failed' and result['answer']['claims'] == []
    trace = json.loads(Path(result['trace_path']).read_text(encoding='utf-8'))
    stage = next(s for s in trace['stages'] if s['stage'] == 'Generate')
    assert stage['claim_audit']['status'] == ('supported' if audit_status == 'supported' else 'rejected')
    assert stage['citation_validation']['semantic_support'] == 'not_evaluated'
    assert base64.b64encode(PNG).decode() not in json.dumps(trace)


def test_requested_figure_mismatch_fails_closed(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    ev = fixture_image(image_root)
    def transport(url, body, headers, timeout):
        return response({"intent":"fact", "standalone_query":"figure", "needs_clarification":False, "paper_ids":["p"], "resolved_entities":{}})
    reasoner = CompatibleReasoner(APIConfig("secret", "fixture", max_retries=0, repair_retries=0), "s", transport, image_root=image_root)
    pipeline = PaperPipeline(TextRetriever(), reasoner, tmp_path/"traces", tmp_path/"memory", mode="bm25", visual_retriever=VisualRetriever(image_root,ev))
    with pytest.raises(ValueError, match="outside the requested figure"):
        pipeline.run("Describe Figure 2", PaperAgentState("s"))


@pytest.mark.parametrize("visual_enabled", [False, True])
def test_text_index_visual_entity_cannot_bypass_controls(tmp_path, visual_enabled):
    image_root = tmp_path / "images"
    image_root.mkdir()
    ev = fixture_image(image_root)
    text = TextRetriever()
    text.by_id[ev['chunk_id']] = ev
    text.search = lambda *args, **kwargs: {"hits":[ev, text.item]}
    class EmptyVisual:
        def __init__(self):
            self.image_root = image_root
            self.evidence = [ev]
        def search_text(self, *args, **kwargs): return {"hits":[]}
        def search_caption(self, *args, **kwargs): return {"hits":[]}
    requests = []
    def transport(url, body, headers, timeout):
        requests.append(body)
        if len(requests) == 1:
            return response({"intent":"fact", "standalone_query":"figure", "needs_clarification":False, "paper_ids":["p"], "resolved_entities":{}})
        return response({"decision":"refuse", "reason":"insufficient"})
    reasoner = CompatibleReasoner(APIConfig("secret", "fixture", max_retries=0, repair_retries=0), "s", transport, image_root=image_root if visual_enabled else None)
    visual = EmptyVisual() if visual_enabled else None
    pipeline = PaperPipeline(text, reasoner, tmp_path/"traces", tmp_path/"memory", mode="bm25", visual_retriever=visual)
    assert pipeline.run("Describe Figure 2", PaperAgentState("s"))["answer"]["decision"] == "refuse"
    content = requests[1]["messages"][1]["content"]
    assert isinstance(content, str) and 'visual-1' not in content


def test_model_image_echo_is_rejected_without_logging(tmp_path):
    ev = fixture_image(tmp_path)
    def transport(url, body, headers, timeout):
        return response({"decision":"refuse", "reason":"data:image/png;base64," + base64.b64encode(PNG).decode()})
    reasoner = CompatibleReasoner(APIConfig("secret", "fixture", max_retries=0, repair_retries=0), "s", transport, image_root=tmp_path)
    with pytest.raises(RuntimeError):
        reasoner.structured("reflect", {"question":"x"}, images=[ev])
    assert base64.b64encode(PNG).decode() not in json.dumps(reasoner.calls)
