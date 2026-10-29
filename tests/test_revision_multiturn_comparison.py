import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from paper_agent.memory_manager import MemoryManager
from paper_agent.pipeline import PaperAgentState


def module():
    spec=importlib.util.spec_from_file_location('revision_multiturn_comparison',
                                                 Path('scripts/revision_multiturn_comparison.py'))
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def rows():
    return [{'source_paper_id':p,'query_id':'q'+p,'query':'test question '+p,
             'gold_units':[p+'::x']} for p in ('paperA','paperB','paperA','paperC')]


def test_fixed_six_cases_and_budgets():
    m=module(); seeds,cases=m.fixed_cases(rows())
    assert [s['paper_id'] for s in seeds]==['paperC','paperB']
    assert [c['id'] for c in cases]==['simple','comparison','time-exclusion','one-side-missing','recent','singular']
    assert [c['id'] for c in cases if c['needs_history']]==['recent','singular']
    assert all([r['source_paper_id'] for r in c['targets']]==['paperA','paperB'] for c in cases[-2:])
    spec=m.protocol(seeds,cases)
    assert spec['max_actual_requests_total']==36 and spec['max_actual_requests_per_turn']==3
    assert spec['max_output_tokens_per_request']==2048 and spec['max_possible_output_tokens']==73728
    assert spec['transport_retries']==spec['schema_repairs']==spec['judge_requests']==0


def test_actual_history_is_recovered_from_isolated_sqlite(tmp_path):
    m=module()
    seed=PaperAgentState('history')
    seed.turn_count=2
    seed.messages=[{'role':'user','content':'paperC','turn_id':'s1'},
                   {'role':'assistant','content':{'decision':'answer','claims':[]},'turn_id':'s1'},
                   {'role':'user','content':'paperB','turn_id':'s2'},
                   {'role':'assistant','content':{'decision':'answer','claims':[]},'turn_id':'s2'}]
    seed.committed_turns={'s1':{'question':'paperC','result':{'answer':{'decision':'answer'}}},
                          's2':{'question':'paperB','result':{'answer':{'decision':'answer'}}}}
    before=copy.deepcopy(seed)
    hashes=[]
    for name in ('baseline','split'):
        manager=MemoryManager(tmp_path/name,None,None,share_project=False)
        hashes.append(m.seed_sqlite_copy(manager,seed))
        restored=manager.load_working_memory(PaperAgentState('history'))
        assert restored.messages==before.messages and restored.committed_turns==before.committed_turns
        assert restored.memory_namespace==manager.identity
        assert manager.store.snapshot('other') is None
        with pytest.raises(ValueError): m.seed_sqlite_copy(manager,seed)
    assert hashes[0]==hashes[1] and seed==before


def test_summary_unknown_and_evidence_are_not_semantic_truth():
    m=module()
    _,cases=m.fixed_cases(rows())
    trace={'commit_status':'committed','budget':{'used':{'subqueries':1}},'stages':[
        {'stage':'NLU','status':'ok','output':{'paper_ids':['paperA'],'standalone_query':'q'}},
        {'stage':'Rerank','output':{'evidence_ids':['chunk']}}]}
    retriever=SimpleNamespace(by_id={'chunk':{'paper_id':'paperA','spans':[{'paragraph_id':'x'}]}})
    reasoner=SimpleNamespace(calls=[])
    item={}
    m.summarize_attempt(item,trace,reasoner,0,__import__('time').perf_counter(),retriever,cases[0])
    assert item['target_resolution_match'] and item['paper_coverage']['paperA']
    assert item['official_gold_paragraph_hit']['qpaperA']
    assert item['answer_completeness'] is item['unsupported_conclusions'] is item['false_refusal'] is None


def test_dry_run_does_not_create_output_or_read_credentials(tmp_path):
    output=tmp_path/'new'
    run=subprocess.run([sys.executable,'scripts/revision_multiturn_comparison.py','--output',str(output),'--dry-run'],
                       capture_output=True,text=True)
    assert run.returncode==0 and 'max_actual_requests_total' in run.stdout
    assert not output.exists()


def test_full_fixed_fixture_has_bounded_calls_and_paired_sqlite_history(tmp_path,monkeypatch):
    from paper_agent.providers import APIConfig
    m=module()
    chunks=[{'paper_id':p,'chunk_id':p+'-chunk','text':'document text '+p,
             'spans':[{'paragraph_id':'x'}]} for p in ('paperA','paperB','paperC')]
    class Retriever:
        def __init__(self,*_):
            self.chunks=chunks
            self.by_id={c['chunk_id']:c for c in chunks}
            self.metadata={'corpus_hash':'fixture'}
            self.reranker=None
        def search(self,query,mode,top_k,candidate_k,paper_ids):
            hits=[c for c in chunks if paper_ids is None or c['paper_id'] in paper_ids][:top_k]
            return {'hits':hits,'candidate_ids':[c['chunk_id'] for c in hits]}
    class Reasoner:
        def __init__(self,config,session_id): self.config=config;self.calls=[]
        def structured(self,stage,payload):
            if len(self.calls)>=self.config.max_calls: raise RuntimeError('fixture budget reached')
            self.calls.append({'stage':stage,'status':'ok','usage_complete':False})
            if stage=='nlu':
                q=payload['question']
                papers=[p for p in ('paperA','paperB','paperC') if p in q]
                if '刚才' in q or '它（' in q: papers.append('paperB')
                return {'intent':'compare' if len(papers)>1 else 'fact','standalone_query':q,
                        'paper_ids':papers,'resolved_entities':{},'needs_clarification':False}
            if stage=='reflect': return {'decision':'answer'}
            hit=payload['evidence'][0]
            return {'claims':[{'text':hit['text'],'citations':[{'paper_id':hit['paper_id'],'chunk_id':hit['chunk_id']}]}]}
    monkeypatch.setattr(m,'read_jsonl',lambda _:rows())
    monkeypatch.setattr(m,'Retriever',Retriever)
    monkeypatch.setattr(m,'CompatibleReasoner',Reasoner)
    monkeypatch.setattr(m.APIConfig,'from_env',classmethod(lambda cls: APIConfig('fixture','fixture')))
    # Earlier conservative clarifications now consume fewer calls. Keep a
    # deliberately small fixture budget to exercise the exhausted path.
    monkeypatch.setattr(m,'MAX_REQUESTS',28)
    out=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['revision_multiturn_comparison.py','--output',str(out)])
    m.main()
    import json
    result=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert result['shared_history_ready'] and result['actual_requests_total']==28
    assert len(result['seed_turns'])==2 and len(result['runs'])==12
    recent=[r for r in result['runs'] if r['case_id']=='recent']
    assert len(recent)==2 and all(r['status']=='completed' for r in recent)
    assert recent[0]['shared_history_hash']==recent[1]['shared_history_hash']==result['shared_history_hash']
    assert all(r['status']=='not_attempted_budget_exhausted' for r in result['runs'] if r['case_id']=='singular')
