from paper_agent.decomposition import scope_guard
from paper_agent.pipeline import PaperAgentState

def test_reference_without_explicit_paper_must_be_grounded():
    catalog=[{'paper_id':'paperA','title':'Paper A'}, {'paper_id':'paperB','title':'Paper B'}]
    state=PaperAgentState('s')
    state.memory.summary['discussed_papers']=['paperA']
    # A list of retrieved papers in an old summary is not proof of discussion.
    assert scope_guard('刚才那篇说了什么？',catalog,['paperA'],state)=='explicit_targets_unresolved'
    state.messages=[{'role':'user','turn_id':'old','content':'讨论 paperA'}]
    assert scope_guard('刚才那篇说了什么？',catalog,['paperA'],state) is None
    assert scope_guard('刚才那篇说了什么？',catalog,['paperB'],state)=='explicit_targets_unresolved'
    assert scope_guard('刚才那篇说了什么？',catalog,None,state)=='explicit_targets_unresolved'
