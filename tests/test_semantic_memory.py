import copy
import pytest

from paper_agent.pipeline import PaperAgentState
from paper_agent.compaction import semantic_compact_state
from paper_agent.memory import ContextArchive


def state():
    s=PaperAgentState('s')
    for i in range(6):
        claim={'text':f'fact-{i}','citations':[{'paper_id':f'p{i}','chunk_id':f'c{i}'}]}
        s.messages.extend([{'role':'user','content':f'question-{i}'},
                           {'role':'assistant','content':{'claims':[claim]}}])
        s.memory.update({'it':f'p{i}'},[{'key':str(i),'text':f'fact-{i}','evidence_ids':[f'c{i}']}],[],[f'c{i}'])
    return s


def test_semantic_selection_keeps_facts_exact_and_archives_history(tmp_path):
    s=state()
    before=copy.deepcopy(s)
    class Reasoner:
        def structured(self,stage,payload):
            assert stage=='memory_summary'
            return {'overview':'Topics p0 through p5'}
    out=semantic_compact_state(s,tmp_path,Reasoner())
    assert s.memory.summary['paper_order'][0]=='p0'
    assert s.memory.facts['0']==before.memory.facts['0']
    assert len(s.messages)==8
    assert len(ContextArchive(tmp_path).read(out['archive_id'])['messages'])==12


def test_semantic_summary_rejects_invented_facts_before_mutation(tmp_path):
    s=state(); before=copy.deepcopy(s)
    class Reasoner:
        def structured(self,stage,payload):
            return {'overview':123}
    with pytest.raises(ValueError):
        semantic_compact_state(s,tmp_path,Reasoner())
    assert s==before
