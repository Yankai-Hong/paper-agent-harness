"""Validation of local, page-derived evidence. No image bytes enter records."""
import hashlib
import math
import re
from pathlib import Path


def composite_provenance(source):
    """Project reviewed panel provenance to bounded, content-free locations."""
    if (source.get('association_method') != 'explicit_page_reviewed_composite'
            or source.get('review_status') != 'agent_reviewed_not_human'):
        return {}
    details, ids = source.get('source_region_details'), source.get('source_regions')
    if not isinstance(details,list) or not isinstance(ids,list) or not 2 <= len(details) == len(ids) <= 16:
        return {}
    safe = []
    fields = ('paper_id','chunk_id','page','bbox','page_size','page_rotation','coordinate_system',
              'pdf_sha256','image_sha256','panel_label','panel_label_source','panel_scope')
    for item in details:
        if not isinstance(item,dict) or item.get('paper_id') != source.get('paper_id') or item.get('page') != source.get('page') or item.get('pdf_sha256') != source.get('pdf_sha256'):
            return {}
        if not isinstance(item.get('chunk_id'),str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,120}',item['chunk_id']):
            return {}
        box, size = item.get('bbox'), item.get('page_size')
        if (not isinstance(box,(list,tuple)) or len(box)!=4 or not isinstance(size,(list,tuple)) or len(size)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in (*box,*size))
                or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])
                or not re.fullmatch(r'[0-9a-f]{64}',str(item.get('image_sha256','')))):
            return {}
        safe.append({k:item[k] for k in fields if k in item})
    if [r['chunk_id'] for r in safe] != ids:
        return {}
    return {'association_method':'explicit_page_reviewed_composite', 'review_status':'agent_reviewed_not_human',
            'source_regions':ids, 'source_region_details':safe}


def validate_image(evidence, root):
    """Return a verified image path inside root; reject tampering and invalid geometry."""
    root = Path(root).resolve(strict=True)
    relative = evidence.get("image_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Image path must be relative to the evidence root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Image path escapes evidence root")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Unsupported image format")
    if not evidence.get("paper_id") or not evidence.get("chunk_id"):
        raise ValueError("Image requires a paper and chunk identity")
    if evidence.get("evidence_type") not in {"figure", "table"}:
        raise ValueError("Not visual evidence")
    page = evidence.get("page")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("Invalid page")
    bbox, size = evidence.get("bbox"), evidence.get("page_size")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Missing bbox or page size")
    values = [*bbox, *size]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Invalid coordinates")
    x0, y0, x1, y1 = bbox
    w, h = size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Bounding box is outside its page")
    if evidence.get("coordinate_system") != "top-left-points" or evidence.get("page_rotation") not in (0, 90, 180, 270):
        raise ValueError("Unsupported coordinate system or rotation")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != evidence.get("image_sha256"):
        raise ValueError("Image checksum mismatch")
    return path
