"""Render the frozen v2 original PDFs and reproduce their image queries.

PDFs must already be acquired from ACL Anthology and are checksum verified.
Output is created exclusively; original PDFs and existing artifacts are never
modified. This utility does not run Docling, retrieval, or a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture', default='data/eval/multimodal-holdout-v2/cases.json')
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf-8'))
    for paper in fixture['papers']:
        path = Path(args.pdf_root) / (paper['paper_id'] + '.pdf')
        if sha(path) != paper['pdf_sha256']:
            raise ValueError('PDF hash mismatch: ' + paper['paper_id'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    needed = {(q['paper_id'], q['target_page']) for q in fixture['image_queries']}
    for paper_id, pageno in sorted(needed):
        doc = pdfium.PdfDocument(Path(args.pdf_root)/(paper_id + '.pdf'))
        doc[pageno-1].render(scale=1.5).to_pil().save(out / f'{paper_id}-p{pageno}.png')
    for q in fixture['image_queries']:
        page = Image.open(out / f"{q['paper_id']}-p{q['target_page']}.png")
        crop = page.crop(tuple(q['source_page_pixels_at_108dpi']))
        factor = q['resize']
        crop = crop.resize((round(crop.width*factor),round(crop.height*factor)), Image.Resampling.BICUBIC)
        target = out / (q['id'] + '.png')
        crop.save(target)
        if sha(target) != q['query_sha256']:
            raise ValueError('Query checksum mismatch; check pypdfium2/Pillow versions: ' + q['id'])
        print(q['id'], q['query_sha256'])


if __name__ == '__main__':
    main()
