import json
import pytest

from paper_agent.local_audit import LocalClaimAudit
from paper_agent.pipeline import PaperPipeline, PaperAgentState
from paper_agent.providers import APIConfig, CompatibleReasoner


def response(value):
    return {'model':'fixture','choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}],
            'usage':{'prompt_tokens':1,'completion_tokens':1}}


@pytest.mark.parametrize('repair_mode', ['add','changed_text','wrong_id','no_addition'])
def test_numeric_repair_only_adds_trusted_citations_and_rechecks(tmp_path, repair_mode):
    common = {'paper_id':'p','evidence_type':'text','pdf_sha256':'a'*64,'page':1,
              'page_size':[100,100],'coordinate_system':'top-left-points','page_rotation':0}
    first = {**common,'chunk_id':'first','bbox':[1,1,50,20],'text':'There are 10,000 sentences.'}
    second = {**common,'chunk_id':'second','bbox':[1,25,50,50],'text':'It takes 5 seconds.'}
    class Text:
        by_id = {'first':first,'second':second}
        metadata = {'corpus_hash':'test'}
        reranker = None
        def search(self, *args, **kwargs):
            return {'hits':[second,first]}
    class Visual:
        image_root = tmp_path
        evidence = []
        def search_text(self,*args,**kwargs): return {'hits':[]}
        def search_caption(self,*args,**kwargs): return {'hits':[]}
    calls = []
    text = 'The 10,000 sentences take 5 seconds.'
    def transport(url, body, headers, timeout):
        calls.append(body)
        n = len(calls)
        if n == 1:
            value = {'intent':'fact','standalone_query':'How long?','needs_clarification':False,
                     'paper_ids':['p'],'resolved_entities':{}}
        elif n == 2:
            value = {'decision':'answer','reason':'evidence available'}
        elif n == 3:
            value = {'claims':[{'key':'k','text':text,'citations':[{'paper_id':'p','chunk_id':'second'}]}]}
        elif n == 4:
            cites = [{'paper_id':'p','chunk_id':'second'}]
            if repair_mode != 'no_addition':
                cites.append({'paper_id':'p','chunk_id':'wrong' if repair_mode == 'wrong_id' else 'first'})
            value = {'claims':[{'key':'k','text':'Changed text' if repair_mode == 'changed_text' else text,'citations':cites}]}
        else:
            value = {'verdicts':[{'claim_key':'k','status':'supported'}]}
        return response(value)
    reasoner = CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0), 's', transport,
                                  image_root=tmp_path)
    sink = LocalClaimAudit(tmp_path/'claims','secret')
    pipe = PaperPipeline(Text(),reasoner,tmp_path/'traces',tmp_path/'memory',mode='bm25',
                         visual_retriever=Visual(),visual_claim_check=True,claim_audit_sink=sink)
    result = pipe.run('How long?',PaperAgentState('s'))
    saved = json.loads(next((tmp_path/'claims').glob('*.json')).read_text(encoding='utf-8'))
    assert saved['generated']['claims'][0]['citations'] == [{'paper_id':'p','chunk_id':'second'}]
    assert saved['citation_repair']['response']['claims'][0]['key'] == 'k'
    if repair_mode == 'add':
        assert result['answer']['decision'] == 'answer' and len(calls) == 5
        assert {r['chunk_id'] for r in result['answer']['claims'][0]['citations']} == {'first','second'}
        assert saved['verdicts'][0]['status'] == 'supported'
        assert saved['citation_repair']['status'] == 'accepted_after_self_check'
    else:
        assert result['answer']['reason_code'] == 'numeric_citation_incomplete' and len(calls) == 4
        assert saved['citation_repair']['status'] == 'rejected'
