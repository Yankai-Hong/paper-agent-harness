from paper_agent.eval_tasks import memory_metrics

def test_retired_fact_checks_are_unknown_not_false():
    rows=[{'id':'a','checkpoints':{'entities':{'it':'p'},'facts':{'old':'x'},'fact_source_papers':['p']}}]
    result=memory_metrics(rows,{'a':{'entities':{'it':'p'},'legacy_fact_metrics':'unavailable'}})
    assert result['checked_keys']==1 and result['unknown_checks']==2 and result['unknown_cases']==1
    assert [c['correct'] for c in result['per_check']]==[None,True,None]
    assert result['checkpoint_accuracy']['correct']==0
