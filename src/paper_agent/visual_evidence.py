"""Validation of local, page-derived evidence. No image bytes enter records."""
import hashlib
import math
from pathlib import Path


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
