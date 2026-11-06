"""Render and reproduce frozen v3 public PDF query crops without retrieval.

Acquire the PDFs from their official ACL HTTPS URLs before running. Output
directory creation is exclusive, and existing data are never overwritten.
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
    parser.add_argument('--fixture', default='data/eval/multimodal-holdout-v3/cases.json')
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf8'))
    for paper in fixture['papers']:
        path = Path(args.pdf_root) / (paper['paper_id'] + '.pdf')
        if sha(path) != paper['pdf_sha256']:
            raise ValueError('PDF hash mismatch: ' + paper['paper_id'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    needed = {(q['paper_id'],q['target_page']) for q in fixture['image_queries']}
    for pid,page in sorted(needed):
        doc = pdfium.PdfDocument(Path(args.pdf_root)/(pid+'.pdf'))
        doc[page-1].render(scale=1.5).to_pil().save(out/f'{pid}-p{page}.png')
    for query in fixture['image_queries']:
        page = Image.open(out/f"{query['paper_id']}-p{query['target_page']}.png")
        crop = page.crop(tuple(query['source_page_pixels_at_108dpi']))
        factor = query['resize']
        crop = crop.resize((round(crop.width*factor),round(crop.height*factor)), Image.Resampling.BICUBIC)
        target = out/(query['id']+'.png')
        crop.save(target)
        if sha(target) != query['query_sha256']:
            raise ValueError('Query checksum mismatch: '+query['id'])
        print(query['id'], query['query_sha256'])


if __name__ == '__main__':
    main()
