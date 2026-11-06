from paper_agent.pipeline import _frontmatter_candidates, _private_visual_trace


def row(chunk, text, bbox):
    return {'paper_id':'p','chunk_id':chunk,'page':1,'bbox':bbox,'page_size':[600,800],
            'coordinate_system':'top-left-points','page_rotation':0,'pdf_sha256':'a'*64,
            'title':'A sufficiently long paper title','text':text,'evidence_type':'text'}


def test_frontmatter_is_located_and_requires_unique_explicit_scope():
    title=row('t','A sufficiently long paper title',[50,50,500,80])
    authors=row('a','Jane Smith, John Doe',[80,100,480,120])
    assert [r['chunk_id'] for r in _frontmatter_candidates('title and author', 'p', {'t':title,'a':authors})] == ['t','a']
    assert _frontmatter_candidates('title and author', None, {'t':title,'a':authors}) == []
    assert _frontmatter_candidates('title', 'p', {'t':title,'a':authors}) == []
    assert _frontmatter_candidates('title and author', 'p', {'t':title,'a':{**authors,'pdf_sha256':'b'*64}}) == []
    assert _frontmatter_candidates('title and author', 'p', {'t':title,'a':authors,'duplicate':{**title,'chunk_id':'dup'}}) == []


def test_private_trace_exposes_only_known_budget_codes():
    assert _private_visual_trace({'budget_exhausted':'turn_subqueries_exhausted'})['budget_exhausted'] == 'turn_subqueries_exhausted'
    assert 'budget_exhausted' not in _private_visual_trace({'budget_exhausted':'private sensitive data'})
