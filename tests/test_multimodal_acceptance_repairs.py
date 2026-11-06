import pytest
import json

from paper_agent.decomposition import explicit_natural_metadata_scope, unknown_literal_metadata_id
from paper_agent.pipeline import (_apply_context_repairs, _attach_table_number_citations,
                                  _expand_located_context, _expand_verified_associations, _repair_collection_conditions)


def test_natural_frontmatter_scope_requires_same_unique_id_and_no_history():
    catalog=[{'paper_id':'2020.acl-main.703','title':'BART paper'}, {'paper_id':'P18-1031','title':'ULMFiT paper'}]
    assert explicit_natural_metadata_scope('论文 2020.acl-main.703 的完整标题和首位作者是谁？', catalog, ['2020.acl-main.703']) == '2020.acl-main.703'
    for q, ids in [('论文 P99-9999 的标题和作者？',['2020.acl-main.703']),
                   ('论文 2020.acl-main.703 和 P18-1031 的标题和作者？',['2020.acl-main.703','P18-1031']),
                   ('之前那篇论文 2020.acl-main.703 的标题和作者？',['2020.acl-main.703']),
                   ('论文 2020.acl-main.703 的标题和作者？',['P18-1031'])]:
        assert explicit_natural_metadata_scope(q,catalog,ids) is None
    assert unknown_literal_metadata_id('论文 P99-9999 的标题和作者？',catalog)
    assert not unknown_literal_metadata_id('论文 2020.acl-main.703 的标题和作者？',catalog)


@pytest.mark.parametrize('model_papers', [['P99-9999'], None])
def test_unknown_paper_metadata_question_clarifies_without_retrieval(tmp_path, model_papers):
    from paper_agent.pipeline import PaperPipeline, PaperAgentState
    from paper_agent.observability import Exporter
    class Text:
        by_id={'c':{'paper_id':'P18-1031','chunk_id':'c','title':'Known paper title','text':'abstract'}}
        metadata={'corpus_hash':'test'}; reranker=None
        def search(self,*args,**kwargs): raise AssertionError('Unknown paper must not be searched')
    class Model:
        calls=[]
        def structured(self,stage,payload,images=None):
            assert stage == 'nlu'
            return {'intent':'fact','standalone_query':'paper title and authors','needs_clarification':False,
                    'paper_ids':model_papers,'resolved_entities':{}}
    pipe=PaperPipeline(Text(),Model(),tmp_path/'traces',tmp_path/'memory',mode='bm25',exporter=Exporter(enabled=False))
    result=pipe.run('论文 P99-9999 的标题和作者是什么？',PaperAgentState('s'))
    assert result['answer']['decision'] == 'clarify'


def entries():
    common={'paper_id':'p','page':1,'pdf_sha256':'a'*64,'page_size':[100,100],
            'page_rotation':0,'coordinate_system':'top-left-points'}
    cap={**common,'chunk_id':'c','bbox':[1,60,90,70],'evidence_type':'text','docling_label':'text',
         'text':'Table 1: Test error rates (%).'}
    table={**common,'chunk_id':'t','bbox':[1,10,90,50],'evidence_type':'table','figure_number':'Table 1',
           'caption':cap['text'],'caption_source_chunk_ids':['c'],'caption_association_status':'agent_page_reviewed_not_human',
           'caption_geometry':{'caption_bbox':cap['bbox'],'visual_bbox':[1,10,90,50]}}
    return table,cap


def test_reviewed_caption_is_independently_expanded_and_added():
    t,c=entries()
    evidence,ids=_expand_located_context([t],{'c':c})
    assert ids == ['c']
    claims=[{'key':'k','text':'Table 1 reports 4.6%.','citations':[{'paper_id':'p','chunk_id':'t'}]}]
    result=_attach_table_number_citations(claims,evidence)
    assert result['unknown'] == [] and claims[0]['citations'][-1]['chunk_id'] == 'c'


def test_caption_relation_survives_removal_of_visual_caption_text():
    t,c=entries()
    clean,_=_expand_verified_associations([t,c],{'c':c})
    assert 'caption' not in clean[0]
    claims=[{'key':'k','text':'Table 1 reports a value.','citations':[{'paper_id':'p','chunk_id':'t'}]}]
    assert not _attach_table_number_citations(claims,clean,[t])['unknown']
    assert claims[0]['citations'][-1]['chunk_id'] == 'c'
    altered={**t,'bbox':[2,10,90,50]}
    claims=[{'key':'k','text':'Table 1 reports a value.','citations':[{'paper_id':'p','chunk_id':'t'}]}]
    assert _attach_table_number_citations(claims,clean,[altered])['unknown'] == ['k']


def test_docling_reference_is_resolved_only_when_unique_and_same_page():
    t,c=entries()
    t.update(caption_association_status='docling_link_unverified', caption_source_refs=['#/texts/4'])
    c.update(docling_ref='#/texts/4',docling_label='caption')
    evidence,ids=_expand_located_context([t],{'c':c})
    claims=[{'key':'k','text':'Table 1 reports a value.','citations':[{'paper_id':'p','chunk_id':'t'}]}]
    assert ids == ['c'] and not _attach_table_number_citations(claims,evidence)['unknown']
    t2={**t,'caption_source_refs':['#/texts/other']}
    claims=[{'key':'k','text':'Table 1 reports a value.','citations':[{'paper_id':'p','chunk_id':'t'}]}]
    assert _attach_table_number_citations(claims,[t2,c])['unknown'] == ['k']


@pytest.mark.parametrize('key,value',[('pdf_sha256','b'*64),('page',2),('text','Table 2: Different.'),('bbox',[0,0,0,0])])
def test_untrusted_caption_not_attached(key,value):
    t,c=entries(); c[key]=value
    evidence,ids=_expand_located_context([t],{'c':c})
    assert not ids
    claims=[{'key':'k','text':'Table 1 reports 4.6%.','citations':[{'paper_id':'p','chunk_id':'t'}]}]
    assert _attach_table_number_citations(claims,evidence)['unknown'] == ['k']


def test_context_repair_requires_exact_cited_source_quote_and_preserves_other_words():
    _,c=entries(); c['text']='most similar pair among 10,000 sentences takes 5 seconds.'
    claim={'key':'k','text':'SBERT reduces this effort to 5 seconds.','citations':[{'paper_id':'p','chunk_id':'c'}]}
    quote='most similar pair among 10,000 sentences'
    fix={'claim_key':'k','replacement':'the time for finding the '+quote,'source_quote':quote,'paper_id':'p','chunk_id':'c'}
    revised=_apply_context_repairs([claim],[c],[fix],['k'])
    assert revised[0]['text'] == 'SBERT reduces the time for finding the '+quote+' to 5 seconds.'
    assert claim['text'] == 'SBERT reduces this effort to 5 seconds.'
    for bad in ({**fix,'source_quote':'invented quote with enough length'}, {**fix,'chunk_id':'other'}):
        with pytest.raises(ValueError): _apply_context_repairs([claim],[c],[bad],['k'])


def test_collection_repair_requires_located_matching_source_and_self_check_path():
    _, source=entries()
    source['text']='Finding the most similar pair in a collection of 10,000 sentences takes 5 seconds.'
    claim={'key':'k','text':'Finding the most similar pair takes 5 seconds.',
           'citations':[{'paper_id':'p','chunk_id':'c'}]}
    question='How long to find the most similar pair among 10,000 sentences?'
    fixed,report=_repair_collection_conditions(question,[claim],[source])
    assert not report['unknown'] and report['added'][0]['source_quote']=='in a collection of 10,000 sentences'
    assert '10,000 sentences' in fixed[0]['text'] and claim['text'] == 'Finding the most similar pair takes 5 seconds.'
    unrelated={**source,'text':'In a collection of 10,000 sentences, another task takes 5 seconds.'}
    unchanged,report=_repair_collection_conditions(question,[claim],[unrelated])
    assert report['unknown']==['k'] and unchanged==[claim]
    invalid={**source,'bbox':[0,0,0,0]}
    assert _repair_collection_conditions(question,[claim],[invalid])[1]['unknown']==['k']


@pytest.mark.parametrize('verdict', ['supported','unsupported'])
def test_collection_repair_keeps_original_and_requires_verdict(tmp_path, verdict):
    from paper_agent.local_audit import LocalClaimAudit
    from paper_agent.pipeline import PaperPipeline, PaperAgentState
    from paper_agent.providers import APIConfig, CompatibleReasoner
    _, source=entries()
    source['text']='Finding the most similar pair in a collection of 10,000 sentences takes 5 seconds.'
    class Text:
        by_id={'c':source}; metadata={'corpus_hash':'test'}; reranker=None
        def search(self,*args,**kwargs): return {'hits':[source]}
    class Visual:
        evidence=[]; image_root=tmp_path
        def search_text(self,*args,**kwargs): return {'hits':[]}
        def search_caption(self,*args,**kwargs): return {'hits':[]}
    original='Finding the most similar pair takes 5 seconds.'
    values=[{'intent':'fact','standalone_query':'How long?','needs_clarification':False,'paper_ids':['p'],'resolved_entities':{}},
            {'decision':'answer','reason':'available'},
            {'claims':[{'key':'k','text':original,'citations':[{'paper_id':'p','chunk_id':'c'}]}]},
            {'verdicts':[{'claim_key':'k','status':verdict}]}]
    calls=[]
    def transport(url,body,headers,timeout):
        value=values[len(calls)]; calls.append(value)
        return {'model':'fixture','choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1}}
    reasoner=CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0),'s',transport,image_root=tmp_path)
    sink=LocalClaimAudit(tmp_path/'claims','secret')
    pipe=PaperPipeline(Text(),reasoner,tmp_path/'traces',tmp_path/'memory',mode='bm25',
                       visual_retriever=Visual(),visual_claim_check=True,claim_audit_sink=sink)
    result=pipe.run('How long to find the most similar pair among 10,000 sentences?',PaperAgentState('s'))
    saved=json.loads(next((tmp_path/'claims').glob('*.json')).read_text(encoding='utf8'))
    assert saved['generated']['claims'][0]['text'] == original
    assert saved['collection_repair']['added'][0]['source_quote'] == 'in a collection of 10,000 sentences'
    assert result['answer']['decision'] == ('answer' if verdict == 'supported' else 'refuse')


@pytest.mark.parametrize('mode', ['supported','bad_quote','unsupported'])
def test_context_repair_requires_grounding_and_self_check(tmp_path, mode):
    from paper_agent.local_audit import LocalClaimAudit
    from paper_agent.pipeline import PaperPipeline, PaperAgentState
    from paper_agent.providers import APIConfig, CompatibleReasoner
    quote='most similar pair among 10,000 sentences'
    _, source=entries()
    source['text']=quote+' takes 5 seconds.'
    class Text:
        by_id={'c':source}; metadata={'corpus_hash':'test'}; reranker=None
        def search(self,*args,**kwargs): return {'hits':[source]}
    class Visual:
        evidence=[]; image_root=tmp_path
        def search_text(self,*args,**kwargs): return {'hits':[]}
        def search_caption(self,*args,**kwargs): return {'hits':[]}
    values=[{'intent':'fact','standalone_query':'How long?','needs_clarification':False,'paper_ids':['p'],'resolved_entities':{}},
            {'decision':'answer','reason':'available'},
            {'claims':[{'key':'k','text':'SBERT reduces this effort to 5 seconds.',
                        'citations':[{'paper_id':'p','chunk_id':'c'}]}]},
            {'repairs':[{'claim_key':'k','replacement':'the time for finding the '+quote,
                         'source_quote':'a fabricated quotation' if mode == 'bad_quote' else quote,
                         'paper_id':'p','chunk_id':'c'}]},
            {'verdicts':[{'claim_key':'k','status':'unsupported' if mode == 'unsupported' else 'supported'}]}]
    calls=[]
    def transport(url,body,headers,timeout):
        value=values[len(calls)]; calls.append(value)
        return {'model':'fixture','choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}],
                'usage':{'prompt_tokens':1,'completion_tokens':1}}
    reasoner=CompatibleReasoner(APIConfig('secret','fixture',max_retries=0,repair_retries=0), 's',transport,image_root=tmp_path)
    sink=LocalClaimAudit(tmp_path/'claims','secret')
    pipe=PaperPipeline(Text(),reasoner,tmp_path/'traces',tmp_path/'memory',mode='bm25',
                       visual_retriever=Visual(),visual_claim_check=True,claim_audit_sink=sink)
    result=pipe.run('How long?',PaperAgentState('s'))
    saved=json.loads(next((tmp_path/'claims').glob('*.json')).read_text(encoding='utf8'))
    assert saved['generated']['claims'][0]['text'] == 'SBERT reduces this effort to 5 seconds.'
    if mode == 'supported':
        assert result['answer']['decision'] == 'answer' and saved['context_repair']['status'] == 'accepted_after_self_check'
        assert saved['verdicts'][0]['status'] == 'supported' and quote in result['answer']['claims'][0]['text']
    else:
        assert result['answer']['decision'] == 'refuse'
