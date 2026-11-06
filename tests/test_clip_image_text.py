import json
from pathlib import Path

import numpy as np
import pytest

from paper_agent.clip_text_index import build_clip_text_index, ClipTextRetriever
from paper_agent.pipeline import PaperPipeline, PaperAgentState, PaperQuery, _interleave_evidence
from paper_agent.query_images import validate_query_image
from paper_agent.storage import write_jsonl


def source(cid='t', text='alpha beta'):
    return {'paper_id':'p', 'chunk_id':cid, 'evidence_type':'text', 'text':text, 'page':1,
            'bbox':[1,2,3,4], 'page_size':[10,10], 'coordinate_system':'top-left-points', 'page_rotation':0}


class Encoder:
    provenance = {'image':{'id':'aligned-image'}, 'text':{'id':'aligned-text'}}
    python = 'fixture-python'
    text_model_path = 'fixture-model'
    def encode_texts(self, texts):
        return np.array([[0,1] if 'needle' in t else [1,0] for t in texts], dtype='float32')
    def encode_images(self, images):
        return np.array([[0,1] for _ in images], dtype='float32')


def test_index_preserves_long_text_tail_and_source_identity(tmp_path, monkeypatch):
    text = 'alpha ' * 150 + 'needle is in the tail'
    row = source(text=text)
    chunks = tmp_path/'chunks.jsonl'; write_jsonl(chunks, [row])
    def split(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text(encoding='utf8'))
        Path(request['output']).write_text(json.dumps({'spans':[[[0, 750], [700,len(text)]]],
            'limit':128, 'window':126, 'overlap':16}), encoding='utf8')
    monkeypatch.setattr('paper_agent.clip_text_index.subprocess.run', split)
    meta = build_clip_text_index(chunks, tmp_path/'idx', Encoder())
    retriever = ClipTextRetriever(tmp_path/'idx', Encoder(), meta['source_hash'])
    hit = retriever.search_vector([[0,1]])['hits'][0]
    assert hit['chunk_id'] == 't' and hit['text'] == text and 'needle' in hit['matched_text']
    assert hit['matched_span'] == [700,len(text)]
    with pytest.raises(ValueError, match='dimension'):
        retriever.search_vector([[1,2,3]])
    with pytest.raises(ValueError, match='version'):
        ClipTextRetriever(tmp_path/'idx', Encoder(), 'wrong')
    class Wrong(Encoder): provenance = {'image':{},'text':{}}
    with pytest.raises(ValueError, match='provenance'):
        ClipTextRetriever(tmp_path/'idx', Wrong())
    with pytest.raises(ValueError, match='overwrite'):
        build_clip_text_index(chunks, tmp_path/'idx', Encoder())


def test_image_validation_and_independent_selection(tmp_path):
    from PIL import Image
    path = tmp_path/'user.png'; Image.new('RGB', (3,3)).save(path)
    info = validate_query_image(path)
    assert info['input_type'] == 'user_image' and 'paper_id' not in info and 'chunk_id' not in info
    path.write_bytes(b'not an image')
    with pytest.raises(ValueError): validate_query_image(path)
    a, b = source('a'), source('b')
    assert [x['chunk_id'] for x in _interleave_evidence([a], [b, a])] == ['a','b']


def test_caption_kind_filters_real_text_entities_before_top_k(tmp_path, monkeypatch):
    rows = [source('body','needle body'), {**source('caption','needle caption'), 'docling_label':'caption'}]
    chunks=tmp_path/'chunks.jsonl'; write_jsonl(chunks,rows)
    def split(command,**kwargs):
        request=json.loads(Path(command[-1]).read_text(encoding='utf8'))
        Path(request['output']).write_text(json.dumps({'spans':[[[0,len(t)]] for t in request['inputs']],
            'limit':128,'window':126,'overlap':16}),encoding='utf8')
    monkeypatch.setattr('paper_agent.clip_text_index.subprocess.run',split)
    build_clip_text_index(chunks,tmp_path/'idx',Encoder())
    retriever=ClipTextRetriever(tmp_path/'idx',Encoder())
    hits=retriever.search_vector([[0,1]],1,kind='caption')['hits']
    assert hits[0]['chunk_id']=='caption' and hits[0]['text']=='needle caption'
    assert retriever.search_vector([[0,1]],5,kind='text')['hits'][0]['chunk_id']=='body'
    with pytest.raises(ValueError,match='kind'): retriever.search_vector([[0,1]],kind='image')


@pytest.mark.parametrize('question', ['', 'What does the image show?'])
def test_query_modes_route_images_without_turning_input_into_evidence(tmp_path, question):
    from PIL import Image
    path = tmp_path/'user.png'; Image.new('RGB', (3,3)).save(path)
    item = source()
    class Text:
        metadata = {'corpus_hash':'v'}; by_id = {'t':item}; reranker = None
        def search(self, *args, **kwargs): return {'hits':[item]}
    class Visual:
        image_root = tmp_path; evidence = []; encoder = Encoder()
        def search_text(self, *a, **k): return {'hits':[]}
        def search_caption(self, *a, **k): return {'hits':[]}
        def _search(self, *a, **k): return {'hits':[]}
    class ClipText:
        metadata = {'source_hash':'v'}
        def search_vector(self, *a, **k): return {'hits':[item]}
    calls = []
    class Reasoner:
        def structured(self, stage, payload, images=None):
            calls.append((stage,payload,images))
            if stage == 'nlu': return {'intent':'fact','standalone_query':question,'paper_ids':None,'needs_clarification':False,'resolved_entities':{}}
            if stage == 'reflect': return {'decision':'answer'}
            return {'claims':[{'key':'a','text':'alpha beta','citations':[{'paper_id':'p','chunk_id':'t'}]}]}
    pipe = PaperPipeline(Text(), Reasoner(), tmp_path/'traces', tmp_path/'memory', mode='bm25',
                         visual_retriever=Visual(), clip_text_retriever=ClipText(), archive_recall=False)
    result = pipe.run_query(PaperQuery(question,(str(path),)), PaperAgentState('s'),turn_id='t')
    assert result['answer']['decision'] == ('answer' if question else 'refuse')
    if not question:
        assert result['answer']['reason_code'] == 'image_match_uncertain'
    for stage, payload, images in calls:
        if stage in {'reflect','generate'}:
            assert images[0]['input_type']=='user_image'
            assert all(e['chunk_id'] == 't' for e in payload['evidence'])
    trace = json.loads(Path(result['trace_path']).read_text(encoding='utf8'))
    assert trace['query_mode'] == ('text_image' if question else 'image')
    routes = next(x for x in trace['stages'] if x['stage']=='Rerank')['output']['routes']
    image_routes = {r['route']:r['status'] for r in routes if r['route'].startswith('image_')}
    assert image_routes['image_to_text']=='ok' and image_routes['image_to_image']=='empty'
    if not question: assert image_routes['image_anchor']=='failed'
    assert str(path) not in json.dumps(trace)


def test_missing_image_index_rejected(tmp_path):
    from PIL import Image
    path = tmp_path/'user.png'; Image.new('RGB', (3,3)).save(path)
    class Text: metadata={}; by_id={}
    pipe = PaperPipeline(Text(), object(), tmp_path/'t', tmp_path/'m', mode='bm25')
    with pytest.raises(ValueError, match='both CLIP'):
        pipe.run('', PaperAgentState('s'), images=[str(path)])


def test_pipeline_rejects_mismatched_text_and_model_versions(tmp_path):
    class Text: metadata={'corpus_hash':'v'}; by_id={}
    class Visual: image_root=tmp_path; evidence=[]; encoder=Encoder()
    class WrongCorpus: metadata={'source_hash':'old'}
    with pytest.raises(ValueError, match='corpus versions'):
        PaperPipeline(Text(),object(),tmp_path/'t',tmp_path/'m',mode='bm25',visual_retriever=Visual(),clip_text_retriever=WrongCorpus())
    class WrongEncoder(Encoder): provenance = {'image':'other','text':'other'}
    class WrongModel: metadata={'source_hash':'v'}; encoder=WrongEncoder(); sources=[]
    with pytest.raises(ValueError, match='model versions'):
        PaperPipeline(Text(),object(),tmp_path/'t',tmp_path/'m',mode='bm25',visual_retriever=Visual(),clip_text_retriever=WrongModel())


def test_provider_marks_query_image_as_non_citable(tmp_path):
    from PIL import Image
    from paper_agent.providers import CompatibleReasoner, APIConfig
    path = tmp_path/'query.png'; Image.new('RGB',(3,3)).save(path)
    sent = []
    def transport(url, body, headers, timeout):
        sent.append(body)
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'decision':'refuse','reason':'none'})}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1}}
    reasoner = CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0), 's', transport, image_root=tmp_path)
    reasoner.structured('reflect', {'question':'q','evidence':[]}, images=[validate_query_image(path)])
    blocks = sent[0]['messages'][1]['content']
    assert json.loads(blocks[1]['text']) == {'input_type':'user_image','citation_eligible':False}
    assert blocks[2]['image_url']['url'].startswith('data:image/png;base64,')
    assert 'base64' not in json.dumps(reasoner.calls)


def test_failed_image_text_branch_is_observable(tmp_path):
    from PIL import Image
    path = tmp_path/'query.png'; Image.new('RGB',(3,3)).save(path)
    class Text:
        metadata={'corpus_hash':'v'}; by_id={}; reranker=None
    class Visual:
        image_root=tmp_path; evidence=[]; encoder=Encoder()
        def _search(self,*a,**kw): return {'hits':[]}
    class Clip:
        metadata={'source_hash':'v'}
        def search_vector(self,*a,**kw): raise ValueError('failure')
    class Reasoner:
        def structured(self, stage, payload, images=None): return {'decision':'refuse','reason':'no evidence'}
    pipe = PaperPipeline(Text(),Reasoner(),tmp_path/'traces',tmp_path/'memory',mode='bm25',
                         visual_retriever=Visual(),clip_text_retriever=Clip(),archive_recall=False)
    result = pipe.run('',PaperAgentState('s'),images=[str(path)])
    trace = json.loads(Path(result['trace_path']).read_text(encoding='utf8'))
    assert trace['degraded'] is True
    routes = next(s for s in trace['stages'] if s['stage']=='Rerank')['output']['routes']
    assert {r['route']:r['status'] for r in routes} == {'image_to_text':'failed','image_to_image':'empty','image_anchor':'failed'}


def test_near_duplicate_gate_fails_closed_and_deduplicates():
    from paper_agent.pipeline import _near_duplicate_anchors
    a={'paper_id':'p','chunk_id':'a','visual_score':.96}
    b={'paper_id':'q','chunk_id':'b','visual_score':.75}
    assert _near_duplicate_anchors([[a,b],[a,b]],2,1)==[a]
    assert _near_duplicate_anchors([[a,{**b,'visual_score':.94}]],1,2) is None
    assert _near_duplicate_anchors([[{**a,'visual_score':.89},b]],1,2) is None
    assert _near_duplicate_anchors([[a]],1,2) is None


def test_pure_image_keeps_only_anchor_and_its_located_caption(tmp_path):
    import hashlib
    from PIL import Image
    query=tmp_path/'query.png'; Image.new('RGB',(3,3),'red').save(query)
    image_sha=hashlib.sha256(query.read_bytes()).hexdigest(); pdf_sha='a'*64
    caption={**source('caption','Figure 1: alpha beta'), 'docling_label':'caption','pdf_sha256':pdf_sha}
    unrelated={**source('other','Unrelated candidate'), 'paper_id':'q'}
    figure={'paper_id':'p','chunk_id':'fig','evidence_type':'figure','text':'Figure 1: alpha beta',
            'caption':'Figure 1: alpha beta','page':1,'bbox':[1,2,3,4],'page_size':[10,10],
            'coordinate_system':'top-left-points','page_rotation':0,'pdf_sha256':pdf_sha,
            'image_path':'query.png','image_sha256':image_sha,'visual_score':.97}
    other_figure={**figure,'paper_id':'q','chunk_id':'other-fig','visual_score':.72}
    class Text:
        metadata={'corpus_hash':'v'}; by_id={'caption':caption,'other':unrelated}; reranker=None
    class Visual:
        image_root=tmp_path; evidence=[figure,other_figure]; encoder=Encoder()
        def _search(self,*args,**kwargs): return {'hits':[figure,other_figure]}
    class Clip:
        metadata={'source_hash':'v'}
        def search_vector(self,*args,**kwargs): return {'hits':[unrelated]}
    calls=[]
    class Reasoner:
        def structured(self,stage,payload,images=None):
            calls.append((stage,payload,images))
            if stage=='reflect': return {'decision':'answer'}
            return {'claims':[{'key':'k','text':'alpha beta','citations':[{'paper_id':'p','chunk_id':'caption'}]}]}
    pipe=PaperPipeline(Text(),Reasoner(),tmp_path/'traces',tmp_path/'memory',mode='bm25',archive_recall=False,
                       visual_retriever=Visual(),clip_text_retriever=Clip())
    result=pipe.run('',PaperAgentState('s'),images=[str(query)])
    assert result['answer']['decision']=='answer'
    for stage,payload,images in calls:
        assert {e['chunk_id'] for e in payload['evidence']}=={'fig','caption'}
        assert images[0]['input_type']=='user_image' and images[1]['chunk_id']=='fig'
