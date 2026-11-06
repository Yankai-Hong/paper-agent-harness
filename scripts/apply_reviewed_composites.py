"""Apply explicit, page-reviewed composite regions without modifying Docling output."""
import argparse
import hashlib
import json
import math
import re
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
    p.add_argument("--panel-manifest-output", help="New JSONL retaining excluded panel evidence and composite linkage")
    args = p.parse_args()
    root, output = Path(args.output_root), Path(args.output)
    if root.exists() or output.exists():
        raise ValueError("Outputs must be new; original evidence is never overwritten")
    panel_manifest = Path(args.panel_manifest_output) if args.panel_manifest_output else None
    if panel_manifest and panel_manifest.exists():
        raise ValueError("Panel manifest must be a new file")
    entries = [json.loads(line) for file in args.evidence for line in Path(file).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {(e['paper_id'], e['chunk_id']): e for e in entries}
    if len(by_key) != len(entries):
        raise ValueError("Duplicate evidence identity")
    overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
    if overrides.get('review_status') != 'agent_reviewed_not_human':
        raise ValueError('An explicit agent page review is required')
    excluded = {(item['paper_id'], item['chunk_id']) for item in overrides.get('excluded_evidence', [])}
    if not excluded.issubset(by_key):
        raise ValueError("Excluded evidence does not exist in source")
    entries = [entry for entry in entries if (entry['paper_id'], entry['chunk_id']) not in excluded]
    root.mkdir(parents=True, exist_ok=False)
    panel_records = []
    for item in entries:
        if item.get('evidence_type') not in {'figure', 'table'}:
            continue
        path = validate_image(item, args.source_root)
        target = root / item['image_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        validate_image(item, root)
    for spec in overrides.get('caption_overrides', []):
        paper = spec['paper_id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', paper) or paper in {'.', '..'}:
            raise ValueError('Unsafe paper identity')
        visual = by_key[(paper, spec['visual_chunk_id'])]
        caption = by_key[(paper, spec['caption_chunk_id'])]
        if visual not in entries or caption not in entries:
            raise ValueError('Reviewed association must retain both source items')
        if (visual.get('evidence_type') != 'table' or caption.get('evidence_type') != 'text'
                or visual.get('figure_number') is not None or visual.get('caption_association_status') != 'unknown_missing'
                or not re.fullmatch(r'Table \d+[a-z]?', spec['figure_number'], re.I)
                or not re.match(r'^' + re.escape(spec['figure_number']) + r'\s*:', caption.get('text', ''), re.I)):
            raise ValueError('Caption or original visual conflicts with reviewed table')
        if any(visual.get(key) != caption.get(key) for key in ('paper_id', 'page', 'page_size', 'page_rotation', 'coordinate_system', 'pdf_sha256')):
            raise ValueError('Caption and table must share original page geometry')
        if (visual.get('page_rotation') != 0 or visual.get('coordinate_system') != 'top-left-points'
                or visual.get('region_count') != 1 or caption.get('region_count') != 1
                or visual['bbox'] != spec['visual_bbox'] or caption['bbox'] != spec['caption_bbox']
                or visual['pdf_sha256'] != spec['pdf_sha256']):
            raise ValueError('Reviewed source geometry does not match')
        w, h = visual['page_size']
        cb = caption['bbox']
        if len(cb) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in cb) or not (0 <= cb[0] < cb[2] <= w and 0 <= cb[1] < cb[3] <= h):
            raise ValueError('Invalid caption geometry')
        pdf_path = Path(args.pdf_directory) / (paper + '.pdf')
        if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != spec['pdf_sha256']:
            raise ValueError('PDF checksum mismatch')
        visual.update(figure_number=spec['figure_number'], caption=caption['text'],
                      caption_source_refs=[caption['docling_ref']], caption_source_chunk_ids=[caption['chunk_id']],
                      caption_association_status='agent_page_reviewed_not_human',
                      caption_geometry={'method':'explicit_page_review', 'caption_bbox':cb, 'visual_bbox':visual['bbox']})
        visual['text'] = '\n\n'.join(part for part in (caption['text'], visual.get('table_text')) if part)
    for spec in overrides['overrides']:
        paper, page = spec['paper_id'], spec['page']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', paper) or paper in {'.', '..'}:
            raise ValueError('Unsafe paper identity')
        members = [by_key[(paper, cid)] for cid in spec['member_chunk_ids']]
        caption = by_key[(paper, spec['caption_chunk_id'])]
        if len(members) < 2 or len(set(spec['member_chunk_ids'])) != len(members):
            raise ValueError("Composite needs distinct source regions")
        if any(m['page'] != page or m['evidence_type'] != 'figure' or m['pdf_sha256'] != spec['pdf_sha256'] for m in members):
            raise ValueError("Composite members do not match the reviewed source")
        if (caption.get('page') != page or caption.get('pdf_sha256') != spec['pdf_sha256']
                or caption.get('docling_label') != 'caption' or not caption['text'].startswith(spec['figure_number'] + ':')):
            raise ValueError("Composite caption does not match the reviewed figure")
        bbox = spec['bbox']
        w, h = members[0]['page_size']
        if len(bbox) != 4 or spec['coordinate_system'] != 'top-left-points' or not (0 <= bbox[0] < bbox[2] <= w and 0 <= bbox[1] < bbox[3] <= h):
            raise ValueError("Invalid composite geometry")
        if any(not (bbox[0] <= m['bbox'][0] and bbox[1] <= m['bbox'][1] and m['bbox'][2] <= bbox[2] and m['bbox'][3] <= bbox[3]) for m in members):
            raise ValueError("Composite does not contain all reviewed panels")
        if any(m.get('page_size') != members[0].get('page_size') or m.get('page_rotation') != 0
               or m.get('coordinate_system') != spec['coordinate_system'] for m in members):
            raise ValueError("Composite members need matching, unrotated page geometry")
        if panel_manifest:
            # All intersecting same-page visual regions must be reviewed. Do not
            # use nearest-caption geometry to guess ownership of another panel.
            for candidate in by_key.values():
                if (candidate.get('paper_id') != paper or candidate.get('page') != page
                        or candidate.get('evidence_type') != 'figure' or candidate in members):
                    continue
                x0,y0,x1,y1 = candidate['bbox']
                if min(x1,bbox[2]) > max(x0,bbox[0]) and min(y1,bbox[3]) > max(y0,bbox[1]):
                    raise ValueError('Unreviewed visual region intersects the composite')
            if not {(paper, cid) for cid in spec['member_chunk_ids']}.issubset(excluded):
                raise ValueError('Reviewed panels must be excluded from the deduplicated index')
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
        if panel_manifest:
            for member in members:
                source_path = validate_image(member, args.source_root)
                panel_target = root / member['image_path']
                panel_target.parent.mkdir(parents=True, exist_ok=True)
                if not panel_target.exists():
                    shutil.copy2(source_path, panel_target)
                validate_image(member, root)
                panel_records.append({'paper_id':paper, 'chunk_id':member['chunk_id'], 'page':page,
                    'bbox':member['bbox'], 'page_size':member['page_size'], 'page_rotation':member['page_rotation'],
                    'coordinate_system':member['coordinate_system'], 'pdf_sha256':member['pdf_sha256'],
                    'image_path':member['image_path'], 'image_sha256':member['image_sha256'],
                    'original_figure_number':member.get('figure_number'),
                    'original_caption_association_status':member.get('caption_association_status'),
                    'composite_chunk_id':chunk_id, 'figure_number':spec['figure_number'],
                    'panel_label':spec.get('panel_labels',{}).get(member['chunk_id'], 'unknown'),
                    'panel_label_source':'agent_original_page_review',
                    'panel_scope':'partial_region_not_complete_figure'})
            entry['source_region_details'] = [r for r in panel_records if r['composite_chunk_id'] == chunk_id]
        validate_image(entry, root)
        entries.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
    if panel_manifest:
        panel_manifest.parent.mkdir(parents=True, exist_ok=True)
        with panel_manifest.open('x', encoding='utf-8') as stream:
            for record in panel_records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({'evidence': len(entries), 'composites': len(overrides['overrides']),
                      'reviewed_captions': len(overrides.get('caption_overrides', [])), 'excluded': len(excluded)}))


if __name__ == '__main__':
    main()
