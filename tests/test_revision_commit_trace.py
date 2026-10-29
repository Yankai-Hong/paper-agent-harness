from paper_agent.pipeline import PaperAgentState,TurnTrace
from test_reliability import pipeline

def test_trace_write_failure_cannot_mask_committed_result(tmp_path,monkeypatch):
    p=pipeline(tmp_path,mode='bm25')
    state=PaperAgentState('s')
    monkeypatch.setattr(TurnTrace,'finish',lambda *_: (_ for _ in ()).throw(OSError('private path')))
    result=p.run('alpha',state,'t')
    assert result['answer']['decision']=='answer'
    assert result['trace_status']=='write_failed' and result['degraded']
    replay=p.run('alpha',PaperAgentState('s'),'t')
    assert replay['replayed_commit'] and replay['answer']==result['answer']
