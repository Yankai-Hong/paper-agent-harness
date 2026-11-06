import json
from pathlib import Path

from paper_agent.pipeline import PaperPipeline, PaperAgentState


def test_text_to_image_fusion_uses_same_image_ids_once():
    from paper_agent.visual_retrieval import VisualRetriever
    visual = VisualRetriever.__new__(VisualRetriever)
    visual.metadata = {'corpus_hash':'v'}
    a={'paper_id':'p','chunk_id':'a'}; b={'paper_id':'p','chunk_id':'b'}; c={'paper_id':'q','chunk_id':'c'}
    visual.search_text=lambda *args: {'hits':[a,b]}
    visual.search_caption=lambda *args: {'hits':[b,c]}
    hits=visual.search_text_fused('query')['hits']
    assert [x['chunk_id'] for x in hits]==['b','a','c']
    assert hits[0]['clip_rank']==2 and hits[0]['caption_rank']==1


def test_optional_bge_failure_keeps_other_routes_and_is_traced(tmp_path):
    item = {'paper_id':'p','chunk_id':'t','text':'Supported text.','title':'Paper'}
    class Text:
        metadata={}; by_id={'t':item}; reranker=None
        def search(self,*args,**kwargs): return {'hits':[item]}
    class Visual:
        image_root=tmp_path; evidence=[]
        def search_texts(self,queries,**kwargs): return [{'hits':[]} for _ in queries]
        def search_caption(self,query,**kwargs): return {'hits':[]}
    class Semantic:
        def search(self,query,kind,**kwargs):
            if kind == 'caption': raise RuntimeError('unavailable')
            return {'hits':[]}
    class Reasoner:
        def structured(self,stage,payload,images=None):
            if stage=='nlu': return {'intent':'fact','standalone_query':'Supported text','paper_ids':['p'],
                                      'needs_clarification':False,'resolved_entities':{}}
            if stage=='reflect': return {'decision':'answer'}
            return {'claims':[{'key':'k','text':'Supported text.','citations':[{'paper_id':'p','chunk_id':'t'}]}]}
    pipe=PaperPipeline(Text(),Reasoner(),tmp_path/'traces',tmp_path/'memory',mode='bm25',
                       archive_recall=False,visual_retriever=Visual(),visual_text_retriever=Semantic())
    result=pipe.run('Supported text',PaperAgentState('s'))
    assert result['answer']['decision']=='answer'
    trace=json.loads(Path(result['trace_path']).read_text(encoding='utf8'))
    assert trace['degraded'] is True
    routes=next(s for s in trace['stages'] if s['stage']=='Rerank')['output']['routes']
    statuses={r['route']:r['status'] for r in routes if r['route'] in {'caption_bge_original','table_bge_original'}}
    assert statuses=={'caption_bge_original':'failed','table_bge_original':'empty'}
