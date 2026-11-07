import pytest

from paper_agent.page_fusion import fuse_page_rankings, select_page_evidence


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


def test_evidence_fusion_preserves_sources_and_never_joins_unknown_pages():
    def evidence(paper, chunk, page, mapping='verified'):
        return {'paper_id': paper, 'chunk_id': chunk, 'page_id': page, 'page_mapping': mapping}
    lanes = [[evidence('p', 't1', '1'), evidence('p', 't2', '1'),
              evidence('p', 'missing', None, 'unknown')],
             [evidence('p', 'figure', '1'), evidence('other', 'fig2', '1'),
              evidence('p', 'missing', None, 'unknown')]]
    result = select_page_evidence(lanes, [1, 1])
    page = result[0]
    assert (page['paper_id'], page['page_id']) == ('p', '1')
    assert page['score'] == pytest.approx(2 / 11)
    assert [(x['lane'], x['rank'], x['item']['chunk_id']) for x in page['evidence']] == [
        (0, 1, 't1'), (0, 2, 't2'), (1, 1, 'figure')]
    assert len([x for x in result if x['page_mapping'] == 'unknown']) == 2
    assert any(x['paper_id'] == 'other' and x['page_id'] == '1' for x in result)


def test_page_fusion_rejects_nonfinite_parameters():
    with pytest.raises(ValueError):
        fuse_page_rankings([['a']], [float('nan')])
    with pytest.raises(ValueError):
        select_page_evidence([[]], [1], k=float('inf'))
