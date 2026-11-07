import hashlib

from paper_agent.docling_page_evidence import resolve_page_visual_evidence


def test_page_source_resolution_keeps_visual_evidence_separate_and_unknown(tmp_path):
    path = tmp_path / 'figure.png'
    path.write_bytes(b'page crop')
    item = {'paper_id': 'p', 'chunk_id': 'fig', 'pdf_sha256': 'pdf', 'page': 1,
            'evidence_type': 'figure', 'image_path': 'figure.png',
            'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'parser': {'name': 'docling'},
            'bbox': [1, 2, 3, 4], 'page_size': [10, 10], 'coordinate_system': 'top-left-points', 'page_rotation': 0}
    pages = {'p1': {'paper_id': 'p', 'pdf_sha256': 'pdf', 'page_number': 1}}
    result = resolve_page_visual_evidence(['p1'], pages, [item], tmp_path)
    assert result['by_page']['p1'][0]['association_status'] == 'same_page_only_unverified'
    path.write_bytes(b'tampered')
    result = resolve_page_visual_evidence(['p1'], pages, [item], tmp_path)
    assert not result['by_page']['p1'] and result['unknown'][0]['status'] == 'source_validation_failed'
