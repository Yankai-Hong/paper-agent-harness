from paper_agent.decomposition import plan
from paper_agent.budgets import TurnBudget, BudgetExceeded
from paper_agent.pipeline import PaperAgentState
from test_reliability import pipeline
import pytest
import json
from types import SimpleNamespace
from paper_agent.pipeline import PaperPipeline
from paper_agent.providers import APIConfig, CompatibleReasoner
from test_reliability import response

def test_simple_and_scope():
    nlu={'intent':'fact','standalone_query':'q','paper_ids':['a','b']}
    assert len(plan('compare',nlu,False))==1
    assert len(plan('compare',nlu,True))==2
    bad={**nlu,'sub_questions':[{'query':'q','paper_ids':['foreign']},{'query':'q','paper_ids':['b']}]}
    assert len(plan('compare',bad,True))==1
    assert len(plan('q',{**nlu,'paper_ids':['a']},True))==1
    assert all('do not use 2025' in g['query'] for g in plan('do not use 2025',nlu,True))

def test_pipeline_simple_same_calls(tmp_path):
    results=[]
    for enabled in (False,True):
        p=pipeline(tmp_path/str(enabled),mode='bm25',decomposition=enabled)
        assert p.run('alpha',PaperAgentState('s'),'t')['answer']['decision']=='answer'
        trace=json.loads((tmp_path/str(enabled)/'traces/t.json').read_text())
        results.append((trace['budget']['used']['subqueries'], len(trace['retrieval_plan'])))
    assert results==[(1,1),(1,1)]

def test_budget_exhaustion_no_commit(tmp_path):
    p=pipeline(tmp_path,mode='bm25',budget_limits={'candidates':1})
    assert p.run('alpha',PaperAgentState('s'),'t')['answer']['decision']=='answer'
    clock=[0.]
    b=TurnBudget(seconds=1,clock=lambda:clock[0])
    b.take('subqueries',6)
    with pytest.raises(BudgetExceeded): b.take('subqueries')
    clock[0]=2
    with pytest.raises(BudgetExceeded): b.remaining()

def test_multiple_dimensions_single_paper_and_unrestricted():
    proposals=[{'query':'accuracy','paper_ids':['paperA'],'target':'accuracy'},
               {'query':'latency','paper_ids':['paperA'],'target':'latency'}]
    nlu={'intent':'compare','standalone_query':'accuracy and latency','paper_ids':['paperA'],'sub_questions':proposals}
    assert len(plan('accuracy and latency',nlu,True))==2
    assert len(plan('accuracy and latency',{**nlu,'paper_ids':None,'sub_questions':[dict(p,paper_ids=None) for p in proposals]},True))==2
    assert len(plan('accuracy and latency',{**nlu,'paper_ids':None},True))==1

def test_invalid_optional_plan_falls_back_without_extra_call():
    value={'intent':'compare','standalone_query':'compare','paper_ids':['paperA','paperB'],
           'resolved_entities':{},'needs_clarification':False,
           'sub_questions':[{'id':'one','query':'q','paper_ids':['paperA'],'unexpected':'x'}]}
    r=CompatibleReasoner(APIConfig('secret','test'),'s',lambda *args:response(value))
    assert 'sub_questions' not in r.structured('nlu',{})
    assert len(r.calls)==1 and r.calls[0]['sub_questions_discarded']=='schema_invalid'

def make_complex_pipeline(tmp_path, *, nlu=None, search=None, verdict=None, **kwargs):
    chunks=[{'paper_id':'paperA','chunk_id':'c','text':'alpha A'},
            {'paper_id':'paperB','chunk_id':'c','text':'alpha B'},
            {'paper_id':'paperC','chunk_id':'d','text':'alpha C'}]
    searches=[]; calls=[]
    def search_default(query, **opts):
        searches.append((query,opts))
        hits=[c for c in chunks if opts['paper_ids'] is None or c['paper_id'] in opts['paper_ids']]
        return {'hits':hits[:opts['top_k']],'candidate_ids':[c['chunk_id'] for c in hits[:opts['top_k']]]}
    class Reasoner:
        def structured(self,stage,payload):
            calls.append((stage,payload))
            if stage=='nlu':
                return nlu or {'intent':'compare','standalone_query':'compare accuracy',
                               'paper_ids':['paperA','paperB'],'resolved_entities':{}}
            if stage=='reflect': return verdict(payload) if verdict else {'decision':'answer'}
            return {'claims':[{'text':c['text'],'citations':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']}]} for c in payload['evidence']]}
    retriever=SimpleNamespace(metadata={},by_id={str(i):c for i,c in enumerate(chunks)},
                              reranker=None,search=search or search_default)
    kwargs.setdefault('decomposition',True)
    p=PaperPipeline(retriever,Reasoner(),tmp_path/'traces',tmp_path/'memory',mode='bm25',**kwargs)
    return p,searches,calls

def test_scope_disagreement_clarifies_without_search(tmp_path):
    nlu={'intent':'fact','standalone_query':'accuracy','paper_ids':['paperB'],'resolved_entities':{}}
    p,searches,calls=make_complex_pipeline(tmp_path,nlu=nlu)
    result=p.run('仅限论文paperA的准确率',PaperAgentState('s'),'t')
    assert result['answer']['decision']=='clarify'
    assert not searches and [stage for stage,_ in calls]==['nlu']
    trace=json.loads((tmp_path/'traces/t.json').read_text())
    assert next(s for s in trace['stages'] if s['stage']=='NLU')['scope_guard']=='explicit_targets_unresolved'

def test_multiple_papers_pair_identifiers_and_budgets(tmp_path):
    p,searches,calls=make_complex_pipeline(tmp_path,budget_limits={'candidates':4})
    answer=p.run('Compare paperA and paperB without data from 2024',PaperAgentState('s'),'t')['answer']
    assert answer['decision']=='answer'
    assert {(c['citations'][0]['paper_id'],c['citations'][0]['chunk_id']) for c in answer['claims']}=={('paperA','c'),('paperB','c')}
    assert len(searches)==2 and all(opts['candidate_k']==2 for _,opts in searches)
    assert all('without data from 2024' in query for query,_ in searches)
    trace=json.loads((tmp_path/'traces/t.json').read_text())
    assert trace['budget']['used']['subqueries']==2 and trace['budget']['used']['candidates']==2
    assert len(next(payload for stage,payload in calls if stage=='reflect')['candidate_coverage_not_semantic_support']['g1'])==1

def test_missing_target_is_retrieved_only_within_target_scope(tmp_path):
    counts={}; requests=[]
    def search(query,**opts):
        requests.append((query,opts['paper_ids']))
        paper=opts['paper_ids'][0]
        counts[paper]=counts.get(paper,0)+1
        hits=[] if paper=='paperB' and counts[paper]==1 else [{'paper_id':paper,'chunk_id':paper+'c','text':'alpha'}]
        return {'hits':hits,'candidate_ids':[h['chunk_id'] for h in hits]}
    def reflect(payload):
        if payload['attempt']==1: return {'decision':'retrieve_again','next_query':'more data','missing_goal_ids':['g2']}
        return {'decision':'answer'}
    p,_,_=make_complex_pipeline(tmp_path,search=search,verdict=reflect)
    result=p.run('Compare paperA and paperB',PaperAgentState('s'),'t')
    assert result['answer']['decision']=='answer'
    assert [scope for _,scope in requests]==[['paperA'],['paperB'],['paperB']]
    assert 'Compare paperA and paperB' in requests[-1][0]
    trace=json.loads((tmp_path/'traces/t.json').read_text())
    retrieve=[stage for stage in trace['stages'] if stage['stage']=='Retrieve']
    assert retrieve[0]['candidate_coverage_not_semantic_support']['g2']==[]
    assert len(retrieve[1]['candidate_coverage_not_semantic_support']['g2'])==1

def test_missing_target_or_reflect_disabled_refuses(tmp_path):
    def search(query,**opts):
        hits=[] if opts['paper_ids']==['paperB'] else [{'paper_id':'paperA','chunk_id':'c','text':'alpha'}]
        return {'hits':hits,'candidate_ids':[h['chunk_id'] for h in hits]}
    for reflect in (True,False):
        root=tmp_path/str(reflect)
        p,_,calls=make_complex_pipeline(root,search=search,reflect=reflect)
        result=p.run('Compare paperA and paperB',PaperAgentState('s'),'t')
        assert result['answer']['decision']=='refuse'
        assert not any(stage=='generate' for stage,_ in calls)

def test_out_of_scope_retriever_rolls_back(tmp_path):
    def search(query,**opts):
        return {'hits':[{'paper_id':'paperC','chunk_id':'x','text':'bad'}],'candidate_ids':['x']}
    p,_,_=make_complex_pipeline(tmp_path,search=search)
    state=PaperAgentState('s')
    with pytest.raises(ValueError,match='out-of-scope'): p.run('Compare paperA and paperB',state,'t')
    assert state.turn_count==0 and not state.committed_turns

def test_reported_semantic_gap_and_disabled_reflect_cannot_answer(tmp_path):
    p,_,calls=make_complex_pipeline(tmp_path/'gap',verdict=lambda payload:{'decision':'answer','missing_goal_ids':['g1']})
    assert p.run('Compare paperA and paperB',PaperAgentState('s'),'t')['answer']['decision']=='refuse'
    assert not any(stage=='generate' for stage,_ in calls)
    p,_,calls=make_complex_pipeline(tmp_path/'off',reflect=False)
    assert p.run('Compare paperA and paperB',PaperAgentState('s'),'t')['answer']['decision']=='refuse'
    assert not any(stage=='generate' for stage,_ in calls)

def test_candidate_budget_exceeded_before_search_has_no_commit(tmp_path):
    p,searches,_=make_complex_pipeline(tmp_path,budget_limits={'candidates':1})
    state=PaperAgentState('s')
    with pytest.raises(BudgetExceeded): p.run('Compare paperA and paperB',state,'t')
    assert searches==[] and state.turn_count==0
    trace=json.loads((tmp_path/'traces/t.json').read_text())
    assert trace['commit_status']=='uncommitted' and trace['status']=='error'

def test_explicit_disable_overrides_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('PAPER_DECOMPOSITION','true')
    p,searches,_=make_complex_pipeline(tmp_path,decomposition=False)
    assert p.decomposition is False


@pytest.mark.parametrize('enabled',[False,True])
def test_history_reference_accepts_only_grounded_extra_target(tmp_path, enabled):
    p,searches,calls=make_complex_pipeline(tmp_path,decomposition=enabled)
    state=PaperAgentState('s')
    state.messages=[{'role':'user','content':'讨论 paperB 的方法','turn_id':'old'},
                    {'role':'assistant','content':{'claims':[{'text':'B', 'citations':[{'paper_id':'paperB','chunk_id':'c'}]}]},'turn_id':'old'}]
    state.memory.summary['discussed_papers']=['paperB']
    result=p.run('比较 paperA 和刚才讨论的那篇论文',state,'t')
    assert result['answer']['decision']=='answer'
    assert searches
    refs={(c['paper_id'],c['chunk_id']) for c in next(payload for stage,payload in calls if stage=='generate')['evidence']}
    assert all((cite['paper_id'],cite['chunk_id']) in refs for c in result['answer']['claims'] for cite in c['citations'])
    assert {p for p,_ in refs}=={'paperA','paperB'}


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('question,history',[
    ('比较 paperA 和刚才讨论的那篇论文', []),
    ('只使用 paperA 的信息', ['paperB']),
    ('仅限 paperA 和那篇论文', ['paperB']),
    ('比较 paperA，排除 paperB', ['paperB']),
    ('Compare paperA without the paper paperB', ['paperB']),
    ('比较 paperA', ['paperB']),
])
def test_unsupported_or_restricted_scope_clarifies(tmp_path,enabled,question,history):
    p,searches,calls=make_complex_pipeline(tmp_path,decomposition=enabled)
    state=PaperAgentState('s')
    state.memory.summary['discussed_papers']=history
    result=p.run(question,state,'t')
    assert result['answer']['decision']=='clarify'
    assert not searches and not any(stage=='generate' for stage,_ in calls)


@pytest.mark.parametrize('enabled',[False,True])
def test_only_named_paper_with_matching_nlu_is_allowed(tmp_path,enabled):
    nlu={'intent':'fact','standalone_query':'paperA','paper_ids':['paperA'],'resolved_entities':{}}
    p,searches,calls=make_complex_pipeline(tmp_path,nlu=nlu,decomposition=enabled)
    assert p.run('仅限 paperA',PaperAgentState('s'),'t')['answer']['decision']=='answer'
    assert all(opts['paper_ids']==['paperA'] for _,opts in searches)


@pytest.mark.parametrize('enabled',[False,True])
def test_history_does_not_authorize_any_other_catalog_paper(tmp_path,enabled):
    nlu={'intent':'compare','standalone_query':'q','paper_ids':['paperA','paperC'],'resolved_entities':{}}
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu,decomposition=enabled)
    state=PaperAgentState('s')
    state.memory.summary['discussed_papers']=['paperB']
    assert p.run('比较 paperA 和刚才讨论的那篇论文',state,'t')['answer']['decision']=='clarify'
    assert not searches


def test_only_scope_cannot_silently_drop_history_reference(tmp_path):
    nlu={'intent':'compare','standalone_query':'q','paper_ids':['paperA'],'resolved_entities':{}}
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu)
    state=PaperAgentState('s')
    state.memory.summary['discussed_papers']=['paperB']
    assert p.run('仅限 paperA 和刚才讨论的那篇论文',state,'t')['answer']['decision']=='clarify'
    assert not searches
