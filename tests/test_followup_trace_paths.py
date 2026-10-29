import json
from pathlib import Path
from paper_agent.pipeline import PaperAgentState
from paper_agent.observability import Exporter,attach_scores
from test_reliability import pipeline

def test_two_runs_share_trace_directory_without_overwrite(tmp_path):
    exported=[]
    outputs=[]
    for run in ('baseline','split'):
        p=pipeline(tmp_path/run,mode='bm25',exporter=Exporter(True,exported.append,flush_seconds=.5))
        p.trace_dir=tmp_path/'shared-traces'
        result=p.run('alpha',PaperAgentState('same'),'turn',case_id='case',run_id=run)
        replay=p.run('alpha',PaperAgentState('same'),'turn',case_id='case',run_id=run)
        assert replay['replayed_commit'] and replay['trace_path']==result['trace_path']
        outputs.append(result)
    paths=[Path(r['trace_path']) for r in outputs]
    assert paths[0]!=paths[1] and all(path.exists() for path in paths)
    traces=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
    ids=[t['observability_mapping']['langfuse_trace_id'] for t in traces]
    assert len(set(ids))==2
    for path in paths:
        attach_scores(path,{'behavior_match':True},'rule')
    assert all(json.loads(path.read_text(encoding='utf-8'))['scores'][0]['value']==1 for path in paths)
    assert {item['trace_id'] for item in exported}==set(ids)
