"""Reviewed composites keep provenance while indexing the whole figure once."""
import hashlib
import json
import runpy
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def run_case(tmp_path, monkeypatch, *, unreviewed=False, mismatch_caption=False, table_review=None):
    source = tmp_path/'source'; source.mkdir()
    pdfdir = tmp_path/'pdf'; pdfdir.mkdir()
    pdf = pdfdir/'p.pdf'; pdf.write_bytes(b'fake-pdf')
    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    common = {'paper_id':'p', 'page':1, 'page_size':[100,100], 'page_rotation':0,
              'coordinate_system':'top-left-points', 'pdf_sha256':pdf_hash}
    panels=[]
    for name, bbox in [('a',[10,10,35,40]),('b',[40,10,65,40])]:
        path=source/(name+'.png'); Image.new('RGB',(10,10),'white').save(path)
        panels.append({**common,'chunk_id':name,'evidence_type':'figure','title':'Paper',
            'bbox':bbox,'image_path':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parser':{'name':'test'}})
    if unreviewed:
        path=source/'c.png'; Image.new('RGB',(10,10),'white').save(path)
        panels.append({**common,'chunk_id':'c','evidence_type':'figure','title':'Paper',
            'bbox':[70,10,90,40],'image_path':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parser':{'name':'test'}})
    caption={**common,'chunk_id':'cap','evidence_type':'text','docling_label':'caption',
             'bbox':[10,50,90,60],'text':'Figure 2: two panels',
             'pdf_sha256':'wrong' if mismatch_caption else pdf_hash}
    extra = []
    if table_review is not None:
        path=source/'table.png'; Image.new('RGB',(10,10),'white').save(path)
        table={**common,'chunk_id':'table','evidence_type':'table','title':'Paper', 'region_count':1,
               'bbox':[10,65,90,80], 'image_path':path.name, 'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'figure_number':None, 'caption_association_status':'unknown_missing', 'table_text':'| value |', 'parser':{}}
        table_caption={**common,'chunk_id':'table-cap','docling_ref':'#/texts/12','region_count':1,
                       'evidence_type':'text','docling_label':'text','bbox':[10,82,90,90],'text':'Table 1: result'}
        if table_review == 'wrong_page': table_caption['page']=2
        if table_review == 'wrong_number': table_caption['text']='Table 2: result'
        extra=[table,table_caption]
    evidence=tmp_path/'evidence.jsonl'
    evidence.write_text(''.join(json.dumps(x)+'\n' for x in panels+[caption]+extra),encoding='utf8')
    manifest={'review_status':'agent_reviewed_not_human',
        'excluded_evidence':[{'paper_id':'p','chunk_id':name} for name in ('a','b')],
        'overrides':[{'paper_id':'p','page':1,'pdf_sha256':pdf_hash,'figure_number':'Figure 2',
          'caption_chunk_id':'cap','member_chunk_ids':['a','b'],'panel_labels':{'a':'left','b':'right'},
          'bbox':[5,5,95,45],'coordinate_system':'top-left-points'}]}
    if table_review is not None:
        manifest['caption_overrides']=[{'paper_id':'p','pdf_sha256':pdf_hash,'visual_chunk_id':'table',
            'caption_chunk_id':'table-cap','figure_number':'Table 1',
            'visual_bbox':[10,65,90,79] if table_review == 'wrong_bbox' else [10,65,90,80],
            'caption_bbox':[10,82,90,90]}]
        if table_review == 'wrong_hash': manifest['caption_overrides'][0]['pdf_sha256']='wrong'
    override=tmp_path/'manifest.json'; override.write_text(json.dumps(manifest),encoding='utf8')
    class Page:
        def get_size(self): return (100,100)
        def render(self,scale): return types.SimpleNamespace(to_pil=lambda:Image.new('RGB',(200,200),'white'))
    fake=types.ModuleType('pypdfium2'); fake.PdfDocument=lambda path:[Page()]
    monkeypatch.setitem(sys.modules,'pypdfium2',fake)
    script=Path(__file__).resolve().parents[1]/'scripts'/'apply_reviewed_composites.py'
    module=runpy.run_path(str(script),run_name='composite_test')
    output=tmp_path/'out'; evidence_out=output/'evidence.jsonl'; panel_out=output/'panels.jsonl'
    monkeypatch.setattr(sys,'argv',[str(script),str(evidence),'--overrides',str(override),
        '--pdf-directory',str(pdfdir),'--source-root',str(source),'--output-root',str(output/'crops'),
        '--output',str(evidence_out),'--panel-manifest-output',str(panel_out)])
    module['main']()
    return evidence_out, panel_out, output/'crops'


def test_reviewed_panels_preserved_and_composite_indexed_once(tmp_path,monkeypatch):
    evidence, panel_path, root=run_case(tmp_path,monkeypatch)
    rows=[json.loads(x) for x in evidence.read_text(encoding='utf8').splitlines()]
    panels=[json.loads(x) for x in panel_path.read_text(encoding='utf8').splitlines()]
    visuals=[x for x in rows if x['evidence_type']=='figure']
    assert len(visuals)==1 and visuals[0]['figure_number']=='Figure 2'
    assert [p['chunk_id'] for p in panels]==['a','b']
    assert all(p['composite_chunk_id']==visuals[0]['chunk_id'] and (root/p['image_path']).exists() for p in panels)
    assert all(p['panel_scope']=='partial_region_not_complete_figure' for p in panels)


@pytest.mark.parametrize('kwargs',[{'unreviewed':True},{'mismatch_caption':True}])
def test_reviewed_composite_rejects_unreviewed_overlap_or_other_pdf(tmp_path,monkeypatch,kwargs):
    with pytest.raises(ValueError):
        run_case(tmp_path,monkeypatch,**kwargs)


def test_reviewed_table_links_independent_caption(tmp_path, monkeypatch):
    evidence, _, _ = run_case(tmp_path, monkeypatch, table_review='valid')
    rows = [json.loads(x) for x in evidence.read_text(encoding='utf8').splitlines()]
    table = next(x for x in rows if x['chunk_id'] == 'table')
    caption = next(x for x in rows if x['chunk_id'] == 'table-cap')
    assert table['figure_number'] == 'Table 1'
    assert table['caption_source_chunk_ids'] == ['table-cap']
    assert table['caption_association_status'] == 'agent_page_reviewed_not_human'
    assert table['caption_geometry']['caption_bbox'] == caption['bbox']


@pytest.mark.parametrize('case', ['wrong_page', 'wrong_number', 'wrong_bbox', 'wrong_hash'])
def test_reviewed_table_rejects_conflicting_evidence(tmp_path, monkeypatch, case):
    with pytest.raises(ValueError):
        run_case(tmp_path, monkeypatch, table_review=case)
