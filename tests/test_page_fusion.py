import pytest

from paper_agent.page_fusion import fuse_page_rankings


def test_page_fusion_deduplicates_before_assigning_rank():
    single = fuse_page_rankings([["p1", "p2"], ["p2", "p1"]], [1.0, 0.5], k=10)
    repeated = fuse_page_rankings([["p1", "p1", "p2"], ["p2", "p2", "p1"]], [1.0, 0.5], k=10)
    assert single == repeated == ["p1", "p2"]


def test_page_fusion_rejects_bad_configuration_and_skips_disabled_branch():
    assert fuse_page_rankings([["a", "b"], ["b"]], [1, 0]) == ["a", "b"]
    with pytest.raises(ValueError):
        fuse_page_rankings([["a"]], [-1])
    with pytest.raises(ValueError):
        fuse_page_rankings([["a"]], [1, 1])
