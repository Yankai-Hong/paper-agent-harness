"""Convert one PDF using Docling, retaining page-rendered visual evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_key(label):
    """Normalize only explicit Figure/Table labels, without inferring numbering."""
    match = re.fullmatch(r"\s*(fig(?:ure)?\.?|table)\s*(\d+[a-z]?)\s*", label or "", re.IGNORECASE)
    if not match:
        return None
    return ("table" if match.group(1).lower() == "table" else "figure", match.group(2).lower())


def associate_explicit_references(entries):
    """Link prose naming a uniquely identified item; never use proximity alone."""
    visuals = {}
    for item in entries:
        if item.get("evidence_type") in {"figure", "table"}:
            key = _reference_key(item.get("figure_number"))
            if key:
                visuals.setdefault((item["paper_id"], key), []).append(item)
    pattern = re.compile(r"(?i)\b(fig(?:ure)?\.?|table)\s*(\d+[a-z]?)(?![a-z0-9])")
    for item in entries:
        if item.get("evidence_type") != "text" or item.get("docling_label") == "caption":
            continue
        seen = set()
        for match in pattern.finditer(item.get("text", "")):
            key = _reference_key(match.group(0))
            candidates = visuals.get((item["paper_id"], key), [])
            if len(candidates) != 1 or key in seen:
                continue
            seen.add(key)
            candidates[0]["associated_text"].append({
                "paper_id": item["paper_id"], "chunk_id": item["chunk_id"], "page": item["page"],
                "reference_text": match.group(0), "reference_span": [match.start(), match.end()],
                "method": "explicit_figure_or_table_reference", "text": item["text"]})


def export_table_text(item, document):
    """Record parser output as unverified; failures must not invent cells."""
    try:
        value = item.export_to_markdown(doc=document)
        if not isinstance(value, str) or not value.strip():
            return None, "unknown"
        return value.strip(), "docling_extracted_unverified"
    except Exception:
        return None, "unknown"


def table_structure(item):
    """Extract row/column structure without asserting cell-to-page geometry."""
    try:
        data = item.data
        rows, cols = int(data.num_rows), int(data.num_cols)
        if rows <= 0 or cols <= 0 or rows * cols > 100000:
            return None, 'unknown'
        cells = []
        for cell in data.table_cells:
            start_row, end_row = int(cell.start_row_offset_idx), int(cell.end_row_offset_idx)
            start_col, end_col = int(cell.start_col_offset_idx), int(cell.end_col_offset_idx)
            if not (0 <= start_row < end_row <= rows and 0 <= start_col < end_col <= cols):
                return None, 'unknown'
            cells.append({'text':str(cell.text), 'row_start':start_row, 'row_end':end_row,
                          'column_start':start_col, 'column_end':end_col,
                          'column_header':bool(cell.column_header), 'row_header':bool(cell.row_header)})
        return {'rows':rows, 'columns':cols, 'cells':cells, 'cell_page_bbox':'unknown'}, 'docling_extracted_unverified'
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None, 'unknown'


def checked_caption(item, document, page, kind):
    """Accept only Docling-linked, same-page captions whose label is compatible."""
    refs = list(getattr(item, 'captions', None) or [])
    ref_ids = [str(getattr(ref, 'cref', '')) for ref in refs]
    if not refs:
        return '', None, ref_ids, 'unknown_missing'
    try:
        resolved = [ref.resolve(document) for ref in refs]
        if any(not getattr(cap, 'prov', None) or any(p.page_no != page for p in cap.prov) for cap in resolved):
            return '', None, ref_ids, 'unknown_page_mismatch'
        caption = item.caption_text(document).strip()
    except Exception:
        return '', None, ref_ids, 'unknown_unresolvable'
    match = re.match(r'\s*((?:fig(?:ure)?\.?|table)\s*\d+[a-z]?)\b', caption, re.IGNORECASE)
    number = match.group(1) if match else None
    key = _reference_key(number)
    if key and key[0] != kind:
        return '', None, ref_ids, 'unknown_type_conflict'
    return caption, number, ref_ids, 'docling_link_unverified' if number else 'unknown_number'


def text_span(text, charspan):
    """Return a text slice only when Docling's character provenance is valid."""
    if (not isinstance(text, str) or not isinstance(charspan, (tuple, list)) or len(charspan) != 2
            or any(type(x) is not int for x in charspan) or not 0 <= charspan[0] < charspan[1] <= len(text)):
        return '', 'unknown'
    return text[charspan[0]:charspan[1]].strip(), 'docling_charspan_unverified'


def valid_span_sequence(text, spans):
    """Reject invalid spans or missing non-whitespace text before splitting."""
    end = 0
    for span in spans:
        if (not isinstance(span, (tuple, list)) or len(span) != 2
                or any(type(x) is not int for x in span)
                or not 0 <= span[0] < span[1] <= len(text) or span[0] < end
                or text[end:span[0]].strip()):
            return False
        end = span[1]
    return not text[end:].strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--title", help="Verified paper title (defaults to PDF metadata or filename)")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir", help="Local metadata-only trace directory")
    parser.add_argument("--device", default="auto", help="Docling device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--artifacts-path", help="Read pre-downloaded Docling layout and table models from this directory")
    parser.add_argument("--allow-rotated-90", action="store_true", help="Allow experimentally checked 90-degree displayed-page coordinates")
    return parser.parse_args()


def run(args):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.paper_id) or args.paper_id in {".", ".."}:
        raise ValueError("paper-id must be a safe single path component")
    if not re.fullmatch(r"(?:auto|cpu|cuda(?::\d+)?)", args.device):
        raise ValueError("Unsupported device")
    source = Path(args.pdf).resolve(strict=True)
    output, root = Path(args.output), Path(args.image_root).resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing evidence file")
    root.mkdir(parents=True, exist_ok=True)
    paper_dir = root / args.paper_id
    if not paper_dir.resolve().is_relative_to(root):
        raise ValueError("Paper image directory escapes evidence root")
    if paper_dir.exists() and any(paper_dir.iterdir()):
        raise ValueError("Refusing to overwrite an existing paper image directory")
    from pypdf import PdfReader
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, TableItem

    pdf = PdfReader(str(source))
    title = args.title or str((pdf.metadata or {}).get("/Title") or source.stem)
    rotations = [int(p.get("/Rotate", 0)) % 360 for p in pdf.pages]
    if any(r not in ({0, 90} if args.allow_rotated_90 else {0}) for r in rotations):
        raise ValueError("Rotated pages require explicit verified support; only 90 degrees can be enabled")
    options = PdfPipelineOptions()
    options.generate_page_images = True
    options.images_scale = 2.0
    options.do_ocr = False
    options.accelerator_options = AcceleratorOptions(device=args.device)
    if args.artifacts_path:
        options.artifacts_path = Path(args.artifacts_path).resolve(strict=True)
    result = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}).convert(str(source))
    document = result.document
    pdf_hash = sha(source)
    version = importlib.metadata.version("docling")
    entries, skipped_multipage, split_multiregion, unknown_spans = [], 0, 0, 0
    paper_dir.mkdir(parents=True, exist_ok=True)
    for item, _level in document.iterate_items():
        if not getattr(item, "prov", None):
            continue
        is_visual = isinstance(item, (PictureItem, TableItem))
        kind = "table" if isinstance(item, TableItem) else "figure" if is_visual else "text"
        regions = list(item.prov)
        if len(regions) > 1:
            split_multiregion += 1
        group_id = 'g-' + hashlib.sha256(json.dumps([pdf_hash, str(item.self_ref), kind], separators=(',', ':')).encode()).hexdigest()[:24]
        text = str(getattr(item, 'text', '') or '')
        spans_valid = valid_span_sequence(text, [list(r.charspan) for r in regions]) if not is_visual and len(regions) > 1 else True
        for region_index, prov in enumerate(regions):
            page = int(prov.page_no)
            if not 1 <= page <= len(pdf.pages):
                raise ValueError("Docling page number outside PDF")
            page_obj = document.pages[page]
            width, height = float(page_obj.size.width), float(page_obj.size.height)
            box = prov.bbox.to_top_left_origin(page_height=height)
            bbox = [float(box.l), float(box.t), float(box.r), float(box.b)]
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Docling bbox outside page {page}: {bbox}")
            if rotations[page - 1]:
                import pypdfium2 as pdfium
                rendered_size = pdfium.PdfDocument(str(source))[page - 1].get_size()
                if any(abs(a-b) > .5 for a,b in zip(rendered_size, (width, height))):
                    raise ValueError('Rotated page dimensions differ between parser and renderer')
            span = list(prov.charspan)
            if is_visual:
                content, number, refs, caption_status = checked_caption(item, document, page, kind) if len(regions) == 1 else ('', None, [str(getattr(r, 'cref', '')) for r in getattr(item, 'captions', [])], 'unknown_multiregion')
                span_status = 'not_applicable'
            elif len(regions) == 1:
                content, span_status = text.strip(), 'single_region'
            else:
                content, span_status = text_span(text, span) if spans_valid else ('', 'unknown')
                if span_status == 'unknown':
                    unknown_spans += 1
            if not content and not is_visual and len(regions) == 1:
                continue
            identity = [pdf_hash, str(item.self_ref), page, bbox, kind]
            if len(regions) > 1:
                identity += [region_index, span]
            chunk_id = 'v-' + hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:24]
            entry = {'paper_id': args.paper_id, 'chunk_id': chunk_id, 'title': title, 'text': content, 'evidence_type': kind,
                     'page': page, 'bbox': bbox, 'page_size': [width, height], 'coordinate_system': 'top-left-points',
                     'page_rotation': rotations[page - 1], 'pdf_sha256': pdf_hash,
                     'parser': {'name': 'docling', 'version': version, 'images_scale': 2.0, 'device_requested': args.device, 'evidence_schema': 'docling-region-v2'},
                     'docling_ref': str(item.self_ref), 'docling_label': str(getattr(getattr(item, 'label', None), 'value', getattr(item, 'label', 'unknown'))),
                     'region_group_id': group_id, 'region_index': region_index, 'region_count': len(regions),
                     'source_charspan': span, 'source_span_status': span_status}
            if is_visual:
                entry.update(figure_number=number, caption=content, caption_source_refs=refs,
                             caption_association_status=caption_status, associated_text=[],
                             region_relation='unknown_multiregion' if len(regions) > 1 else 'single_region')
                if kind == 'table':
                    table_text, status = export_table_text(item, document) if len(regions) == 1 else (None, 'unknown_multiregion')
                    structure, structure_status = table_structure(item) if len(regions) == 1 else (None, 'unknown_multiregion')
                    # Docling's Markdown serializer may prepend a caption. If
                    # caption provenance was rejected, do not leak that caption
                    # back into searchable table text as if it were verified.
                    if table_text and caption_status in {'unknown_page_mismatch', 'unknown_type_conflict', 'unknown_unresolvable'}:
                        try:
                            rejected = item.caption_text(document).strip()
                            if not rejected or not table_text.startswith(rejected):
                                table_text, status = None, 'unknown_caption_conflict'
                            else:
                                table_text = table_text[len(rejected):].strip() or None
                                if table_text is None:
                                    status = 'unknown_caption_conflict'
                        except Exception:
                            table_text, status = None, 'unknown_caption_conflict'
                    entry.update(table_text=table_text, table_text_status=status, table_structure=structure, table_structure_status=structure_status)
                    if table_text:
                        entry['text'] = table_text if content and table_text.startswith(content) else '\n\n'.join(part for part in (content, table_text) if part)
                # Crop rendered full pages rather than embedded images, preserving vectors.
                image = page_obj.image.pil_image
                px0, py0 = int(x0 / width * image.width), int(y0 / height * image.height)
                px1, py1 = int(x1 / width * image.width + 0.999999), int(y1 / height * image.height + 0.999999)
                filename = chunk_id + '.png'
                image.crop((px0, py0, px1, py1)).convert('RGB').save(paper_dir / filename)
                entry['image_path'] = (Path(args.paper_id) / filename).as_posix()
                entry['image_sha256'] = sha(paper_dir / filename)
            entries.append(entry)
    if not entries:
        raise ValueError("Docling returned no usable evidence")
    associate_explicit_references(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    result = {"paper_id": args.paper_id, "pdf_sha256": pdf_hash, "docling": version,
              "device_requested": args.device,
              "items": len(entries), "visual_items": sum(e["evidence_type"] in {"figure", "table"} for e in entries),
              "skipped_multiple_provenances": skipped_multipage, "split_multiregion_items":split_multiregion,
              "unknown_spans":unknown_spans}
    print(json.dumps(result))
    return result


def main():
    args = parse_args()
    start, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    outcome, error, result = "error", None, None
    try:
        result = run(args)
        outcome = "ok"
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        if args.trace_dir:
            trace_dir = Path(args.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            turn_id = "parse-" + uuid.uuid4().hex
            stage = {"stage": "Parse", "status": outcome, "started_at": started,
                     "ended_at": datetime.now(timezone.utc).isoformat(),
                     "elapsed_ms": (time.perf_counter() - start) * 1000, "error_type": error}
            if result:
                stage["output"] = {k: result[k] for k in ("pdf_sha256", "docling", "device_requested", "items", "visual_items", "skipped_multiple_provenances", "split_multiregion_items", "unknown_spans")}
            trace = {"session_id": "pdf-ingest", "turn_id": turn_id, "stages": [stage]}
            with (trace_dir / (turn_id + ".json")).open("x", encoding="utf-8") as stream:
                json.dump(trace, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
