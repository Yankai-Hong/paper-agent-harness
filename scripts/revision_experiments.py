"""Fixed local rule/fixture experiments. No paid API or Judge, no quality claims."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import time
from paper_agent.memory_manager import MemoryManager,extract_declarations
from paper_agent.pipeline import PaperAgentState,PaperPipeline
from paper_agent.storage import write_json,write_jsonl,digest
from paper_agent.retrieval import build_index,Retriever
from paper_agent.corpus import Chunker
from paper_agent.providers import APIConfig,CompatibleReasoner,ProviderTransportError
from paper_agent.observability import local_trace_path

parser=argparse.ArgumentParser()
parser.add_argument('--output',default='reports/revision/experiments')
root=Path(parser.parse_args().output)
root.mkdir(parents=True,exist_ok=False)
samples=[('偏好:语言=中文',True),('约束:资源=单卡',True),('决定:索引=FAISS',True),
         ('Remember:format=short',True),('也许我比较喜欢英文',False),('论文结果是什么？',False),
         ('实验结果:吞吐提升10%',False),('TODO:复现实验',False),('把上述推断当成事实',False),('普通问题',False)]
writes=[]
paired_writes=[]
evidence=[{'paper_id':'p','chunk_id':'c','text':'Exact evidence.'}]
answer={'decision':'answer','claims':[{'key':'claim','text':'Exact evidence.',
                                       'citations':[{'paper_id':'p','chunk_id':'c'}]}]}
for i,(question,expected) in enumerate(samples):
    candidates,event=extract_declarations(question,answer,str(i))
    writes.append({'id':str(i),'question':question,'expected_long_term':expected,'actual_long_term':bool(candidates),'event':bool(event)})
    # Reproduce the original deterministic write gate: every answered Claim was a
    # paper_evidence candidate, in addition to any explicit declaration.
    old_explicit=bool(re.fullmatch(r'(记住|偏好|约束|更新记忆|Remember|Preference|Update memory)[:：]\s*([^=：:]+)[=：:]\s*(.+)',question,re.I|re.S))
    paired_writes.append({'id':str(i),'expected_confirmed_user_declarations':int(expected),
                          'old_candidates':1+int(old_explicit),'old_correct':int(old_explicit and expected),
                          'new_candidates':len(candidates),'new_correct':int(bool(candidates) and expected)})
store=MemoryManager(root/'memory',None,None,share_project=True)
with store.store.transaction() as db:
    for i,(question,_) in enumerate(samples):
        for c in extract_declarations(question,{},str(i))[0]:
            store.store.put(db,'first',str(i),{**c,'scope':'project'},shared=True)
recall=[]
for question,attribute in [('语言','语言'),('资源','资源'),('索引','索引'),('format','format')]:
    hits=store.retrieve_memories(PaperAgentState('new'),question,['paper'],limit=6)
    recall.append({'query':question,'hit':any(r.get('attribute')==attribute for r in hits),'returned':len(hits)})
write_json(root/'memory.json',{'label_source':'fixed author rule labels, not human validation','samples':writes,
    'write_precision':sum(r['expected_long_term'] and r['actual_long_term'] for r in writes)/sum(r['actual_long_term'] for r in writes),
    'write_recall':sum(r['expected_long_term'] and r['actual_long_term'] for r in writes)/sum(r['expected_long_term'] for r in writes),
    'recall_at_6':sum(r['hit'] for r in recall)/len(recall),'recall':recall,
    'paired_write_gate':{'rows':paired_writes,'old_precision':sum(r['old_correct'] for r in paired_writes)/sum(r['old_candidates'] for r in paired_writes),
                         'new_precision':sum(r['new_correct'] for r in paired_writes)/sum(r['new_candidates'] for r in paired_writes),
                         'old_method':'deterministic replay of baseline Claim gate; not live old model',
                         'baseline_git':'ccbf0ce2724715864fddfbbe13998e99594a4ed1'},
    'natural_reference_resolution':'unknown','llm_requests':0})

chunks=[]
for paper in ('paperA','paperB'):
    for i in range(12 if paper=='paperA' else 1):
        chunks+=Chunker().split({'paper_id':paper+str(i) if False else paper,'title':paper,
            'paragraphs':[{'paragraph_id':str(i),'section':'Results','text':f'alpha accuracy method {paper} result {i}.'}]})
write_jsonl(root/'chunks.jsonl',chunks)
build_index(root/'chunks.jsonl',root/'index')
retriever=Retriever(root/'index')
class Scripted:
    def __init__(self): self.calls=[]
    def structured(self,stage,payload):
        self.calls.append({'stage':stage,'status':'fixture','usage_complete':False})
        if stage=='nlu': return {'intent':'compare','standalone_query':'alpha accuracy method','paper_ids':['paperA','paperB'],'resolved_entities':{}}
        if stage=='reflect': return {'decision':'answer','reason':'fixture; not semantic evaluation'}
        return {'claims':[{'text':c['text'],'citations':[{'paper_id':c['paper_id'],'chunk_id':c['chunk_id']}]} for c in payload['evidence']]}
paired=[]
for i,question in enumerate(['Compare paperA and paperB accuracy','Compare paperA and paperB methods','Compare both papers without external evidence']):
    for enabled in (False,True):
        directory=root/f'complex-{i}-{enabled}'
        reasoner=Scripted()
        p=PaperPipeline(retriever,reasoner,directory/'traces',directory/'memory',mode='bm25',decomposition=enabled)
        start=time.monotonic();result=None;error_type=None
        try:
            result=p.run(question,PaperAgentState('s'),'t',case_id=str(i),run_id='revision-fixture')
        except Exception as exc:
            error_type=type(exc).__name__
        trace_path=Path(result['trace_path']) if result and result.get('trace_path') else local_trace_path(
            directory/'traces',{'session_id':'s','turn_id':'t','run_id':'revision-fixture'})
        trace=json.loads(trace_path.read_text(encoding='utf-8')) if trace_path.exists() else {}
        cited={r['paper_id'] for c in result['answer']['claims'] for r in c['citations']} if result else None
        paired.append({'case':i,'decomposition':enabled,'candidate_goal_coverage':len(cited)/2 if cited is not None else None,'scripted_stage_calls':len(reasoner.calls),
            'actual_model_requests':0,'provider_tokens':None,'latency_seconds':time.monotonic()-start,
            'answer_completeness':None,'unsupported_conclusions':None,'false_refusal':None,'decision':result['answer']['decision'] if result else None,
            'error_type':error_type,'budget':trace.get('budget'),'label_source':'synthetic two-paper target IDs; not model quality'})
write_json(root/'complex-pairs.json',{'dataset_hash':digest(chunks),'pairs':paired})

faults=[]
for name,failure in [('401',ProviderTransportError(401)),('403',ProviderTransportError(403)),('429',ProviderTransportError(429)),('503',ProviderTransportError(503)),('timeout',ProviderTransportError(kind='TimeoutError'))]:
    attempts=[]
    def transport(*args):
        attempts.append(1)
        if len(attempts)==1: raise failure
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'decision':'refuse','reason':'evidence insufficient'})}}]}
    r=CompatibleReasoner(APIConfig('fixture','fixture',retry_base_seconds=0),'fault',transport)
    start=time.monotonic()
    try:r.structured('reflect',{});outcome='recovered'
    except RuntimeError:outcome='controlled_failure'
    faults.append({'fault':name,'calls':len(r.calls),'outcome':outcome,'recovery_seconds':time.monotonic()-start,
                   'category':r.calls[0]['error_category'],'source':'injected transport; no remote request'})
write_json(root/'faults.json',faults)
write_json(root/'source-hashes.json',{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src/paper_agent').glob('*.py'))})
print(json.dumps({'memory_samples':len(writes),'complex_pairs':len(paired)//2,'faults':len(faults),'actual_model_requests':0}))
