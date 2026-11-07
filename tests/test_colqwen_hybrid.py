from types import SimpleNamespace

from paper_agent.colqwen_hybrid import HybridColQwenPageSearch


def test_hybrid_generates_distinct_candidates_and_preserves_unknowns():
    chunks = [dict(chunk_id='a', paper_id='p', page_id='1', text='first'),
              dict(chunk_id='b', paper_id='p', page_id='1', text='second'),
              dict(chunk_id='c', paper_id='p', page_id=None, text='unknown'),
              dict(chunk_id='d', paper_id='p', page_id='2', text='other')]
    text = SimpleNamespace(chunks=chunks, by_id={x['chunk_id']: x for x in chunks},
        search=lambda query, **kwargs: {'candidate_ids': ['a', 'b', 'c', 'd'], 'hits': chunks})
    clip = SimpleNamespace(search=lambda query, limit: {'pages': ['2', '3'], 'search_mode': 'test'})
    sources = {x: {'paper_id': 'p', 'page_number': int(x)} for x in ('1', '2', '3')}
    seen = []
    def search(query, pages):
        seen.extend(pages)
        return {'hits': [{'page_id': page, 'evidence_type': 'page'} for page in reversed(pages)]}
    visual = SimpleNamespace(max_candidates=4, sources=sources, search_text=search,
                             store=SimpleNamespace(contract={'source_hash': 'source'}))
    result = HybridColQwenPageSearch(text, clip, visual, text_source_hash='source', depth=2).search('question')
    assert seen == ['1', '2', '3']
    assert result['candidate_count'] == 3
    assert result['unknown_text_locations'] == [{**chunks[2], 'candidate_rank': 3}]
    assert [x['rank'] for x in result['text_candidate_refs']] == [1, 2, 3, 4]
    assert result['text_candidate_refs'][2]['page_mapping'] == 'unknown'
    assert result['pages'][0]['text_evidence']['chunk_id'] == 'a'
    assert result['pages'][0]['page_evidence']['evidence_type'] == 'page'
