"""Apply explicit, page-reviewed composite regions without modifying Docling output."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import pypdfium2 as pdfium
from paper_agent.visual_evidence import validate_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", nargs="+")
    p.add_argument("--overrides", required=True)
    p.add_argument("--pdf-directory", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root, output = Path(args.output_root), Path(args.output)
    if root.exists() or output.exists():
        raise ValueError("Outputs must be new; original evidence is never overwritten")
    entries = [json.loads(line) for file in args.evidence for line in Path(file).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {(e['paper_id'], e['chunk_id']): e for e in entries}
    if len(by_key) != len(entries):
        raise ValueError("Duplicate evidence identity")
    overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
    excluded = {(item['paper_id'], item['chunk_id']) for item in overrides.get('excluded_evidence', [])}
    if not excluded.issubset(by_key):
        raise ValueError("Excluded evidence does not exist in source")
    entries = [entry for entry in entries if (entry['paper_id'], entry['chunk_id']) not in excluded]
    root.mkdir(parents=True, exist_ok=False)
    for item in entries:
        if item.get('evidence_type') not in {'figure', 'table'}:
            continue
        path = validate_image(item, args.source_root)
        target = root / item['image_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        validate_image(item, root)
    for spec in overrides['overrides']:
        paper, page = spec['paper_id'], spec['page']
        members = [by_key[(paper, cid)] for cid in spec['member_chunk_ids']]
        caption = by_key[(paper, spec['caption_chunk_id'])]
        if len(members) < 2 or len(set(spec['member_chunk_ids'])) != len(members):
            raise ValueError("Composite needs distinct source regions")
        if any(m['page'] != page or m['evidence_type'] != 'figure' or m['pdf_sha256'] != spec['pdf_sha256'] for m in members):
            raise ValueError("Composite members do not match the reviewed source")
        if caption.get('page') != page or caption.get('docling_label') != 'caption' or not caption['text'].startswith(spec['figure_number'] + ':'):
            raise ValueError("Composite caption does not match the reviewed figure")
        bbox = spec['bbox']
        w, h = members[0]['page_size']
        if len(bbox) != 4 or spec['coordinate_system'] != 'top-left-points' or not (0 <= bbox[0] < bbox[2] <= w and 0 <= bbox[1] < bbox[3] <= h):
            raise ValueError("Invalid composite geometry")
        if any(not (bbox[0] <= m['bbox'][0] and bbox[1] <= m['bbox'][1] and m['bbox'][2] <= bbox[2] and m['bbox'][3] <= bbox[3]) for m in members):
            raise ValueError("Composite does not contain all reviewed panels")
        pdf_path = Path(args.pdf_directory) / (paper + '.pdf')
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != spec['pdf_sha256']:
            raise ValueError("PDF checksum mismatch")
        pdf = pdfium.PdfDocument(str(pdf_path))
        page_obj = pdf[page - 1]
        if any(abs(a - b) > 0.5 for a, b in zip(page_obj.get_size(), (w, h))):
            raise ValueError("PDF dimensions differ from Docling")
        image = page_obj.render(scale=2).to_pil().convert('RGB')
        region = (math.floor(bbox[0] / w * image.width), math.floor(bbox[1] / h * image.height),
                  math.ceil(bbox[2] / w * image.width), math.ceil(bbox[3] / h * image.height))
        identity = json.dumps([spec['pdf_sha256'], page, bbox, spec['figure_number'], spec['member_chunk_ids']], separators=(',', ':'))
        chunk_id = 'v-' + hashlib.sha256(identity.encode()).hexdigest()[:24]
        target = root / paper / (chunk_id + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Composite output already exists")
        image.crop(region).save(target)
        entry = {'paper_id': paper, 'chunk_id': chunk_id, 'title': members[0]['title'], 'text': caption['text'],
                 'caption': caption['text'], 'evidence_type': 'figure', 'figure_number': spec['figure_number'],
                 'page': page, 'bbox': bbox, 'page_size': [w, h], 'coordinate_system': spec['coordinate_system'],
                 'page_rotation': 0, 'pdf_sha256': spec['pdf_sha256'], 'image_path': target.relative_to(root).as_posix(),
                 'image_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                 'parser': members[0]['parser'], 'associated_text': [],
                 'source_regions': spec['member_chunk_ids'], 'caption_chunk_id': spec['caption_chunk_id'],
                 'association_method': 'explicit_page_reviewed_composite', 'review_status': overrides['review_status']}
        validate_image(entry, root)
        entries.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(json.dumps({'evidence': len(entries), 'composites': len(overrides['overrides']), 'excluded': len(excluded)}))


if __name__ == '__main__':
    main()
