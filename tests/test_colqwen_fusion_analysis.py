import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('colqwen_fusion_analysis', Path(__file__).parents[1] / 'scripts' / 'colqwen_fusion_analysis.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fusion_counts_one_vote_per_page_and_never_uses_gold_to_rank():
    assert module.ranking(['a', 'a', 'b'], ['b', 'b', 'c'], 1) == ['b', 'a', 'c']
    assert module.ranking(['a', 'b'], ['b'], 0) == ['a', 'b']
    assert module.ranking(['a'], ['missing'], 0) == ['a']


def test_evaluation_keeps_conflicting_label_unknown():
    base = [dict(case_id='s', partition='dev', gold_status='conflict', gold_page='a', hybrid={'pages': ['a']}),
            dict(case_id='v', partition='dev', gold_status='consistent', gold_page='b', hybrid={'pages': ['b']})]
    scored = [dict(case_id='s', partition='dev', gold_status='conflict', group='standard', top20=['a'], text_rank=None, candidate_rank=None),
              dict(case_id='v', partition='dev', gold_status='consistent', group='visual', top20=['b'], text_rank=1, candidate_rank=1)]
    records, metrics = module.evaluate(base, scored, .25)
    assert records[0]['rank'] is None
    assert metrics['standard']['hit1'] == 0 and metrics['visual']['hit1'] == 1
