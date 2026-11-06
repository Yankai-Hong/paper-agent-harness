"""Reproduce the frozen query crop from the public LoRA v2 PDF."""
import argparse
import hashlib
import io
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',required=True); p.add_argument('--fixture-sha256',required=True)
    p.add_argument('--pdf',required=True); p.add_argument('--output',required=True)
    args=p.parse_args()
    raw=Path(args.fixture).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.fixture_sha256: raise ValueError('Fixture hash mismatch')
    fixture=json.loads(raw); pdf=Path(args.pdf); out=Path(args.output)
    if out.exists(): raise ValueError('Refusing to overwrite query image')
    if hashlib.sha256(pdf.read_bytes()).hexdigest()!=fixture['pdf_sha256']: raise ValueError('PDF hash mismatch')
    import pypdfium2 as pdfium
    from PIL import Image
    image=pdfium.PdfDocument(str(pdf))[fixture['page']-1].render(scale=1.5).to_pil()
    crop=image.crop(tuple(fixture['crop_page_pixels_108dpi'])).resize((356,328),Image.Resampling.BICUBIC)
    buffer=io.BytesIO(); crop.save(buffer,format='PNG'); data=buffer.getvalue()
    if hashlib.sha256(data).hexdigest()!=fixture['query_sha256']: raise ValueError('Reproduced crop hash mismatch')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream: stream.write(data)
    print(json.dumps({'output':str(out),'sha256':fixture['query_sha256']}))


if __name__=='__main__': main()
