import importlib.util
from pathlib import Path


spec=importlib.util.spec_from_file_location('irpapers_baseline',Path(__file__).resolve().parents[1]/'scripts/irpapers_baseline.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_page_aggregation_and_metrics_include_failure_and_missing():
    by_id={'a':{'page_id':'p1'},'b':{'page_id':'p1'},'c':{'page_id':'p2'}}
    assert module.page_rank(['a','b','c'],by_id)==['p1','p2']
    items=[{'group':'standard','partition':'dev','gold_page':'p1','question':'Q',
            'x':{'status':'ok','gold_in_index':True,'gold_rank':2}},
           {'group':'standard','partition':'test','gold_page':'missing','question':'Other',
            'x':{'status':'error','gold_in_index':False,'gold_rank':None}},
           {'group':'visual','partition':'dev','gold_page':'p1','question':' Q ',
            'x':{'status':'ok','gold_in_index':True,'gold_rank':2}}]
    scores=module.metrics(items,'x')
    assert scores['standard_all']['denominator']==2 and scores['standard_all']['hit_at_5']==.5
    assert scores['standard_all']['mrr']==.25 and scores['standard_test']['ok']==0
    assert scores['deduplicated_union']['denominator']==2


def test_existing_retriever_hybrid_equivalence(tmp_path):
    import numpy as np
    from paper_agent.retrieval import build_index,Retriever,rrf
    from paper_agent.storage import write_jsonl
    class Encoder:
        model_id='fixture'; max_length=768
        def encode(self,texts):
            return np.array([[1.,0.] if 'alpha' in t else [0.,1.] for t in texts],dtype=np.float32)
    entries=[{'paper_id':'p','page_id':'1','chunk_id':'a','text':'alpha one'},
             {'paper_id':'p','page_id':'2','chunk_id':'b','text':'beta two'},
             {'paper_id':'q','page_id':'3','chunk_id':'c','text':'alpha beta'}]
    source=tmp_path/'chunks.jsonl';write_jsonl(source,entries)
    build_index(source,tmp_path/'index',Encoder()); r=Retriever(tmp_path/'index',Encoder())
    bm=r.search('alpha',mode='bm25',top_k=3,candidate_k=3)['candidate_ids']
    dense=r.search('alpha',mode='dense',top_k=3,candidate_k=3)['candidate_ids']
    expected=[c for c,_ in rrf([bm,dense])]
    assert r.search('alpha',mode='hybrid',top_k=3,candidate_k=3)['candidate_ids']==expected
