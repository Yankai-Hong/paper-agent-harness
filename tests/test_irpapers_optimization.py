import importlib.util
from pathlib import Path


spec=importlib.util.spec_from_file_location('irpapers_optimize',Path(__file__).resolve().parents[1]/'scripts/irpapers_optimize.py')
opt=importlib.util.module_from_spec(spec);spec.loader.exec_module(opt)


def test_page_fusion_deduplicates_and_uses_page_identity():
    hits=opt.ranked([['a','a','b'],['b','c']],[1.,1.])
    assert hits[0]=='b' and set(hits)=={'a','b','c'}
    assert opt.ranked([['a','b'],['b','c']],[1.,0.])==['a','b']


def test_unknown_gold_stays_in_denominator():
    data={'gold_status':'identity_conflict','gold_page':'p'}
    assert opt.rank_gold(['p'],data) is None
    stats=opt.score([1,None])
    assert stats['n']==2 and stats['hit1']==.5 and stats['mrr']==.5
