import importlib.util
from pathlib import Path

def test_comparison_design_is_fixed_and_bounded():
    path=Path('scripts/revision_model_comparison.py')
    spec=importlib.util.spec_from_file_location('revision_model_comparison',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=[{'source_paper_id':p,'query_id':p,'query':'q','gold_units':[p+'::x']} for p in ('p1','p2','p1')]
    cases=module.fixed_cases(rows)
    assert [x['case_id'] for x in cases]==['simple','comparison','missing']
    assert [len(x['targets']) for x in cases]==[1,2,1]
    assert module.MAX_REQUESTS==len(cases)*2*3
    assert module.MAX_OUTPUT_TOKENS<=4096

def test_existing_output_is_never_overwritten(tmp_path):
    import subprocess,sys
    completed=subprocess.run([sys.executable,'scripts/revision_model_comparison.py','--output',str(tmp_path),'--dry-run'],capture_output=True,text=True)
    assert completed.returncode!=0 and 'Output must be a new directory' in completed.stderr
