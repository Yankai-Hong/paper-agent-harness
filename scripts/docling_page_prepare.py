"""Render pages for an existing Docling text index without rebuilding its text."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--text-index', required=True)
    p.add_argument('--pdf-dir', required=True, help='PDF filenames are the exact paper IDs plus .pdf')
    p.add_argument('--output', required=True)
    p.add_argument('--scale', type=float, default=1.5)
    a = p.parse_args()
    if not .5 <= a.scale <= 3:
        raise ValueError('Unsupported page rendering scale')
    import pypdfium2 as pdfium
    from pypdf import PdfReader
    index = Path(a.text_index)
    metadata = json.loads((index / 'metadata.json').read_text(encoding='utf-8'))
    with (index / 'chunks.jsonl').open(encoding='utf-8') as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    corpus = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if corpus != metadata.get('corpus_hash'):
        raise ValueError('Text index corpus checksum mismatch')
    if not chunks or any(x.get('evidence_type') != 'text' or (x.get('parser') or {}).get('name') != 'docling'
                         or not re.fullmatch(r'[A-Za-z0-9_.-]+', str(x.get('paper_id') or ''))
                         or not re.fullmatch(r'[0-9a-f]{64}', str(x.get('pdf_sha256') or ''))
                         or type(x.get('page')) is not int or x['page'] < 1 for x in chunks):
        raise ValueError('Every source chunk must have a located Docling PDF identity')
    papers = sorted({x['paper_id'] for x in chunks})
    pdfs = {}
    for paper in papers:
        path = (Path(a.pdf_dir) / (paper + '.pdf')).resolve(strict=True)
        fingerprint = sha(path)
        if {x['pdf_sha256'] for x in chunks if x['paper_id'] == paper} != {fingerprint}:
            raise ValueError('Docling text and PDF checksum differ: ' + paper)
        reader = PdfReader(str(path))
        if any(int(page.get('/Rotate', 0)) % 360 != 0 for page in reader.pages):
            raise ValueError('Rotated PDF pages are not supported in this preparation')
        if any(x['page'] > len(reader.pages) for x in chunks if x['paper_id'] == paper):
            raise ValueError('Docling page is outside the PDF')
        pdfs[paper] = (path, fingerprint, len(reader.pages))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    records = []
    for paper in papers:
        path, fingerprint, count = pdfs[paper]
        document = pdfium.PdfDocument(str(path))
        for number in range(1, count + 1):
            page_id = f'{paper}__p{number}'
            target = output / 'images' / (page_id + '.png')
            page = document[number - 1]
            bitmap = page.render(scale=a.scale)
            image = bitmap.to_pil().convert('RGB')
            image.save(target, format='PNG')
            width, height = image.size
            image.close(); bitmap.close(); page.close()
            records.append({'page_id': page_id, 'paper_id': paper, 'page_number': number,
                            'pdf_sha256': fingerprint, 'image_path': 'images/' + target.name,
                            'image_sha256': sha(target), 'width': width, 'height': height,
                            'render_scale': a.scale, 'source_kind': 'rendered_pdf_page',
                            'citation_granularity': 'page'})
        document.close()
    if len({x['page_id'] for x in records}) != len(records):
        raise ValueError('Page IDs collide')
    pages = output / 'pages.jsonl'
    with pages.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    binding = {'kind': 'docling_pdf_pages_v1', 'status': 'sealed', 'pages_sha256': sha(pages),
               'text_chunks_sha256': sha(index / 'chunks.jsonl'), 'text_corpus_hash': metadata['corpus_hash'],
               'papers': len(papers), 'pages': len(records), 'render_scale': a.scale,
               'text_provenance': 'existing Docling source; OCR and table parsing require verification'}
    with (output / 'binding.json').open('x', encoding='utf-8') as stream:
        json.dump(binding, stream, ensure_ascii=False, indent=2)
    print(json.dumps(binding, ensure_ascii=False))


if __name__ == '__main__':
    main()
