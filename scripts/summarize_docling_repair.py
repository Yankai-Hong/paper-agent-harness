"""Verify local evidence hashes and write per-page development statistics."""
import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from paper_agent.visual_evidence import validate_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pdf', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--omissions', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    pdf_hash = hashlib.sha256(Path(args.pdf).read_bytes()).hexdigest()
    pages = len(PdfReader(args.pdf).pages)
    entries = [json.loads(x) for x in Path(args.evidence).read_text(encoding='utf8').splitlines() if x.strip()]
    omissions = [json.loads(x) for x in Path(args.omissions).read_text(encoding='utf8').splitlines() if x.strip()]
    counts = {str(i): {'evidence':0, 'visual':0, 'number_unknown':0, 'caption_relation_unknown':0,
                       'quarantined':0, 'skipped_empty':0} for i in range(1, pages+1)}
    for row in entries:
        if row['pdf_sha256'] != pdf_hash or not 1 <= row['page'] <= pages:
            raise ValueError('Evidence PDF identity or page mismatch')
        stats = counts[str(row['page'])]
        stats['evidence'] += 1
        if row['evidence_type'] in {'figure','table'}:
            validate_image(row, args.image_root)
            stats['visual'] += 1
            stats['number_unknown'] += int(not row.get('figure_number'))
            status = row.get('caption_association_status', '')
            reviewed_composite = row.get('association_method') == 'explicit_page_reviewed_composite'
            stats['caption_relation_unknown'] += int(not reviewed_composite and (not status or status.startswith('unknown')))
    missing_page = 0
    for row in omissions:
        if row['pdf_sha256'] != pdf_hash or 'chunk_id' in row:
            raise ValueError('Invalid omission identity')
        if row['page'] is None:
            missing_page += 1
        else:
            stats = counts[str(row['page'])]
            stats['quarantined' if row['disposition'] == 'quarantined' else 'skipped_empty'] += 1
    result = {'pdf_sha256':pdf_hash, 'pages':counts, 'omissions_without_page':missing_page,
              'visual_hashes_checked':sum(p['visual'] for p in counts.values())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
