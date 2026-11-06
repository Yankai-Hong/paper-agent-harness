import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('docling_ingest_geometry', Path(__file__).resolve().parents[1] / 'scripts' / 'docling_ingest.py')
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


@pytest.mark.parametrize(('box', 'reason'), [
    ([1, 2, 1, 3], 'zero_width'), ([1, 2, 3, 2], 'zero_height'),
    ([1, 2, 1, 2], 'zero_width_and_height'), ([3, 2, 1, 4], 'reversed_coordinate'),
    ([1, 4, 3, 2], 'reversed_coordinate'), ([-1, 2, 3, 4], 'outside_page'),
    ([1, 2, 101, 4], 'outside_page'), ([1, float('nan'), 3, 4], 'non_finite_coordinate'),
    ([1, 2, float('inf'), 4], 'non_finite_coordinate'),
    ([1, 2, 1.000001, 2.000001], None), ([0, 0, 100, 200], None),
])
def test_geometry_classification(box, reason):
    assert ingest.bbox_problem(box, 100, 200) == reason


@pytest.mark.parametrize('dimensions', [(0, 200), (100, -1), (float('nan'), 200)])
def test_page_geometry_is_structural_error(dimensions):
    with pytest.raises(ValueError, match='page dimensions'):
        ingest.bbox_problem([1, 2, 3, 4], *dimensions)


def test_bad_region_is_recorded_and_does_not_abort_valid_region(tmp_path, monkeypatch):
    class Box:
        def __init__(self, coords): self.l, self.t, self.r, self.b = coords
        def to_top_left_origin(self, page_height): return self
    def item(ref, coords, text):
        return types.SimpleNamespace(self_ref=ref, text=text, label='text',
            prov=[types.SimpleNamespace(page_no=1, bbox=Box(coords), charspan=[0, len(text)])])
    items = [item('#/texts/bad', [4, 2, 4, 3], 'bad'), item('#/texts/good', [1, 2, 3, 4], 'good')]
    page = types.SimpleNamespace(size=types.SimpleNamespace(width=100, height=200))
    doc = types.SimpleNamespace(pages={1:page}, iterate_items=lambda: ((x, 0) for x in items))
    class Converter:
        def __init__(self, **kwargs): pass
        def convert(self, path): return types.SimpleNamespace(document=doc, status='success')
    class Options: pass
    class Params:
        def __init__(self, **kwargs): pass
    class Pdf:
        def __init__(self, path): self.pages=[{}]; self.metadata={}
    fake_modules = {
        'pypdf': {'PdfReader':Pdf},
        'docling.datamodel.base_models': {'InputFormat':types.SimpleNamespace(PDF='pdf'), 'ConversionStatus':types.SimpleNamespace(SUCCESS='success')},
        'docling.datamodel.pipeline_options': {'PdfPipelineOptions':Options},
        'docling.datamodel.accelerator_options': {'AcceleratorOptions':Params},
        'docling.document_converter': {'DocumentConverter':Converter, 'PdfFormatOption':Params},
        'docling_core.types.doc': {'PictureItem':type('Picture', (), {}), 'TableItem':type('Table', (), {})},
    }
    for name, attributes in fake_modules.items():
        module=types.ModuleType(name)
        for key, value in attributes.items(): setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ingest.importlib.metadata, 'version', lambda name:'test')
    pdf=tmp_path/'p.pdf'; pdf.write_bytes(b'pdf')
    output=tmp_path/'evidence.jsonl'
    args=types.SimpleNamespace(pdf=str(pdf), paper_id='p', title='P', image_root=str(tmp_path/'crops'),
        output=str(output), device='cpu', artifacts_path=None, allow_rotated_90=False)
    result=ingest.run(args)
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    omitted=[json.loads(line) for line in output.with_name(output.name+'.quarantine.jsonl').read_text().splitlines()]
    assert [row['docling_ref'] for row in rows] == ['#/texts/good']
    assert omitted[0]['docling_ref'] == '#/texts/bad' and omitted[0]['reason'] == 'zero_width'
    assert 'chunk_id' not in omitted[0] and omitted[0]['source_charspan'] == [0, 3]
    assert result['page_stats']['1']['quarantined'] == 1 and result['page_stats']['1']['evidence'] == 1
    with pytest.raises(ValueError, match='overwrite'):
        ingest.run(args)
    args.output=str(tmp_path/'other.jsonl')
    args.image_root=str(tmp_path/'other-crops')
    args.paper_id='../escape'
    with pytest.raises(ValueError, match='safe single path component'):
        ingest.run(args)
    args.paper_id='p'
    doc.pages.clear()
    with pytest.raises(ValueError, match='missing PDF page'):
        ingest.run(args)
