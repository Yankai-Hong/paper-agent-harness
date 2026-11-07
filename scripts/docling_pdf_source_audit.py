"""Audit Docling evidence identity and geometry against the original PDFs."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'text-index', 'evidence', 'evidence-root', 'pdf-dir', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    from pypdf import PdfReader
    source, index = Path(a.prepared), Path(a.text_index)
    binding = json.loads((source / 'binding.json').read_text(encoding='utf-8'))
    if binding.get('pages_sha256') != sha(source / 'pages.jsonl') or binding.get('text_chunks_sha256') != sha(index / 'chunks.jsonl'):
        raise ValueError('PDF source or text binding differs')
    pages = rows(source / 'pages.jsonl')
    by_paper = {}
    for page in pages:
        by_paper.setdefault(page['paper_id'], page['pdf_sha256'])
        if by_paper[page['paper_id']] != page['pdf_sha256']:
            raise ValueError('Multiple PDF hashes for one paper')
    sizes = {}
    for paper, fingerprint in by_paper.items():
        path = Path(a.pdf_dir) / (paper + '.pdf')
        if sha(path) != fingerprint:
            raise ValueError('PDF checksum changed: ' + paper)
        reader = PdfReader(str(path))
        for number, page in enumerate(reader.pages, 1):
            if int(page.get('/Rotate', 0)) % 360:
                raise ValueError('Rotated PDF requires a separate audit')
            sizes[(paper, number)] = (float(page.cropbox.width), float(page.cropbox.height))
    evidence = rows(index / 'chunks.jsonl') + [x for x in rows(a.evidence) if x.get('evidence_type') in {'figure', 'table'}]
    results = []
    root = Path(a.evidence_root).resolve()
    for item in evidence:
        paper, number = item.get('paper_id'), item.get('page')
        status = 'ok_geometry'
        size, bbox, expected = item.get('page_size'), item.get('bbox'), sizes.get((paper, number))
        if item.get('pdf_sha256') != by_paper.get(paper) or expected is None:
            status = 'unknown_pdf_or_page'
        elif ((item.get('parser') or {}).get('name') != 'docling' or item.get('coordinate_system') != 'top-left-points'
              or item.get('page_rotation') != 0 or not isinstance(size, (list, tuple)) or len(size) != 2
              or not isinstance(bbox, (list, tuple)) or len(bbox) != 4
              or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in [*size, *bbox])):
            status = 'unknown_geometry'
        elif (any(abs(float(size[i]) - expected[i]) > .1 for i in (0, 1))
              or not (0 <= bbox[0] < bbox[2] <= size[0] and 0 <= bbox[1] < bbox[3] <= size[1])):
            status = 'geometry_mismatch'
        elif item.get('evidence_type') in {'figure', 'table'}:
            relative = item.get('image_path')
            path = (root / relative).resolve() if isinstance(relative, str) and not Path(relative).is_absolute() else None
            if path is None or not path.is_relative_to(root) or not path.is_file() or sha(path) != item.get('image_sha256'):
                status = 'image_source_mismatch'
        results.append({'paper_id': paper, 'chunk_id': item.get('chunk_id'), 'page': number,
                        'evidence_type': item.get('evidence_type'), 'status': status})
    report = {'scope': 'PDF identity, page size, coordinates, and crop checksum; not semantic support',
              'counts': dict(Counter(r['status'] for r in results)), 'results': results}
    with Path(a.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'counts': report['counts'], 'items': len(results)}))


if __name__ == '__main__':
    main()
