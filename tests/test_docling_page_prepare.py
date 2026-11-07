import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('docling_page_prepare', Path(__file__).parents[1] / 'scripts' / 'docling_page_prepare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pdf_page_preparation_binds_text_to_pdf_and_refuses_changed_source(tmp_path, monkeypatch):
    pdf_dir, index = tmp_path / 'pdfs', tmp_path / 'index'
    pdf_dir.mkdir(); index.mkdir()
    pdf = pdf_dir / 'paper.pdf'; pdf.write_bytes(b'original PDF')
    chunk = {'paper_id': 'paper', 'chunk_id': 'c1', 'page': 1, 'evidence_type': 'text',
             'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'parser': {'name': 'docling'}}
    chunks = [chunk]
    corpus = hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (index / 'metadata.json').write_text(json.dumps({'corpus_hash': corpus}))
    (index / 'chunks.jsonl').write_text(json.dumps(chunk) + '\n')
    class Image:
        size = (100, 100)
        def convert(self, mode): return self
        def save(self, path, format): path.write_bytes(b'rendered image')
        def close(self): pass
    class Bitmap:
        def to_pil(self): return Image()
        def close(self): pass
    class Page:
        def render(self, scale): return Bitmap()
        def close(self): pass
        def get(self, key, default): return 0
    class Document:
        def __init__(self, path): pass
        def __getitem__(self, index): return Page()
        def close(self): pass
    monkeypatch.setitem(sys.modules, 'pypdfium2', SimpleNamespace(PdfDocument=Document))
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace(PdfReader=lambda path: SimpleNamespace(pages=[Page()])))
    output = tmp_path / 'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--text-index', str(index), '--pdf-dir', str(pdf_dir), '--output', str(output)])
    module.main()
    page = json.loads((output / 'pages.jsonl').read_text())
    assert page['paper_id'] == 'paper' and page['page_number'] == 1
    assert page['image_sha256'] == hashlib.sha256(b'rendered image').hexdigest()
    pdf.write_bytes(b'changed PDF')
    second = tmp_path / 'second'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--text-index', str(index), '--pdf-dir', str(pdf_dir), '--output', str(second)])
    with pytest.raises(ValueError, match='checksum differ'):
        module.main()
    assert not second.exists()
