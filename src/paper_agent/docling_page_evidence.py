"""Resolve independently located Docling figures and tables on selected pages."""
from .visual_evidence import validate_image


def resolve_page_visual_evidence(page_ids, pages, evidence, image_root):
    selected = set(page_ids)
    locators = {(row.get('paper_id'), row.get('pdf_sha256'), row.get('page_number')): page_id
                for page_id, row in pages.items() if page_id in selected and row.get('pdf_sha256')}
    resolved = {page_id: [] for page_id in selected}
    unknown = []
    seen = set()
    for item in evidence:
        if item.get('evidence_type') not in {'figure', 'table'}:
            continue
        page_id = locators.get((item.get('paper_id'), item.get('pdf_sha256'), item.get('page')))
        if page_id is None:
            continue
        key = (item.get('paper_id'), item.get('chunk_id'))
        if key in seen:
            raise ValueError('Duplicate visual evidence identity')
        seen.add(key)
        try:
            if (item.get('parser') or {}).get('name') != 'docling':
                raise ValueError('Visual evidence is not marked as Docling output')
            validate_image(item, image_root)
        except (OSError, ValueError) as exc:
            unknown.append({'paper_id': item.get('paper_id'), 'chunk_id': item.get('chunk_id'),
                            'page_id': page_id, 'status': 'source_validation_failed', 'error_type': type(exc).__name__})
            continue
        resolved[page_id].append({**item, 'association_status': 'same_page_only_unverified'})
    return {'by_page': resolved, 'unknown': unknown}
