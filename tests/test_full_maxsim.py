import pytest
from paper_agent.full_maxsim import rank_pages


def test_full_ranking_is_descending_and_ties_use_page_id():
    assert rank_pages(['p2','p1','p3'], [1, 1, 2]) == [2, 1, 0]
    with pytest.raises(ValueError): rank_pages(['same','same'], [1, 2])
    with pytest.raises(ValueError): rank_pages(['a'], [float('nan')])
    with pytest.raises(ValueError): rank_pages(['a','b'], [1])
