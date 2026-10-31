import importlib.util
from pathlib import Path


def load_script():
    spec = importlib.util.spec_from_file_location("docling_ingest", Path(__file__).resolve().parents[1] / "scripts" / "docling_ingest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_association_requires_explicit_unambiguous_reference():
    module = load_script()
    figure = {"paper_id": "p", "chunk_id": "f", "evidence_type": "figure", "figure_number": "Figure 1", "associated_text": []}
    table = {"paper_id": "p", "chunk_id": "t", "evidence_type": "table", "figure_number": "Table 1", "associated_text": []}
    text = {"paper_id": "p", "chunk_id": "x", "evidence_type": "text", "page": 2,
            "text": "Fig. 1 shows the model; Table 1 reports results. Fig. 10 differs."}
    other = {"paper_id": "other", "chunk_id": "y", "evidence_type": "text", "page": 1, "text": "Figure 1 is unrelated."}
    module.associate_explicit_references([figure, table, text, other])
    assert len(figure["associated_text"]) == len(table["associated_text"]) == 1
    ref = figure["associated_text"][0]
    assert ref["chunk_id"] == "x" and ref["reference_text"] == "Fig. 1"
    assert text["text"][slice(*ref["reference_span"])] == "Fig. 1"
    duplicate = {**figure, "chunk_id": "f2", "associated_text": []}
    figure["associated_text"] = []
    module.associate_explicit_references([figure, duplicate, text])
    assert not figure["associated_text"] and not duplicate["associated_text"]


def test_table_export_marks_parser_output_unverified_and_errors_unknown():
    module = load_script()
    class Table:
        def export_to_markdown(self, doc):
            assert doc == "document"
            return "| Method | Score |\n|---|---|\n| A | 2 |"
    value, status = module.export_table_text(Table(), "document")
    assert "Score" in value and status == "docling_extracted_unverified"
    class Broken:
        def export_to_markdown(self, doc):
            raise RuntimeError("parser failed")
    assert module.export_table_text(Broken(), "document") == (None, "unknown")


def test_multiregion_text_spans_retain_exact_source_without_overlaps():
    module = load_script()
    text = 'first region second region'
    spans = [[0, 12], [13, len(text)]]
    assert module.valid_span_sequence(text, spans)
    first, status = module.text_span(text, spans[0])
    assert first == 'first region' and status == 'docling_charspan_unverified'
    assert not module.valid_span_sequence(text, [[0, 12], [10, len(text)]])
    assert not module.valid_span_sequence(text, [[0, 100]])
    assert not module.valid_span_sequence(text, [[0, 5], [13, len(text)]])
    assert module.text_span(text, [-1, 5]) == ('', 'unknown')


def test_caption_requires_same_page_and_matching_visual_type():
    module = load_script()
    from types import SimpleNamespace
    class Ref:
        cref = '#/texts/1'
        def __init__(self, page): self.page = page
        def resolve(self, document): return SimpleNamespace(prov=[SimpleNamespace(page_no=self.page)])
    class Visual:
        def __init__(self, page, caption): self.captions, self.caption = [Ref(page)], caption
        def caption_text(self, document): return self.caption
    assert module.checked_caption(Visual(2, 'Figure 1: architecture'), object(), 2, 'figure')[1] == 'Figure 1'
    assert module.checked_caption(Visual(3, 'Figure 1: architecture'), object(), 2, 'figure')[3] == 'unknown_page_mismatch'
    value = module.checked_caption(Visual(2, 'Figure 1: architecture'), object(), 2, 'table')
    assert value[0] == '' and value[1] is None and value[3] == 'unknown_type_conflict'


def test_table_cells_preserve_row_column_and_unknown_geometry():
    module = load_script()
    from types import SimpleNamespace
    cell = SimpleNamespace(start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0,
                           end_col_offset_idx=1, text='75.5', column_header=False, row_header=False)
    item = SimpleNamespace(data=SimpleNamespace(num_rows=1, num_cols=1, table_cells=[cell]))
    grid, status = module.table_structure(item)
    assert status == 'docling_extracted_unverified' and grid['cells'][0]['text'] == '75.5'
    assert grid['cell_page_bbox'] == 'unknown' and grid['cells'][0]['row_start'] == 0
    cell.end_row_offset_idx = 4
    assert module.table_structure(item) == (None, 'unknown')
