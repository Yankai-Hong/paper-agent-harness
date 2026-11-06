"""Run two BART-only image queries after the frozen second-paper parse failure.

These are partial diagnostics in a one-paper index. They are not equivalent to
the frozen two-paper unscoped benchmark and never substitute a target answer.
"""
import json
import time
from pathlib import Path
from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever

fixture=json.loads(Path('data/eval/multimodal-holdout-v3/cases.json').read_text(encoding='utf8'))
encoder=SubprocessEncoder('.venv-mm-20261030/Scripts/python.exe',
    'models/multimodal-v1/clip-image-v1','models/multimodal-v1/clip-text-v1','cuda:0')
visual=VisualRetriever('indexes/multimodal-holdout-v3-bart-clip',
    'data/raw/multimodal-holdout-v3-docling/crops',encoder)
fields=('paper_id','chunk_id','evidence_type','page','bbox','figure_number','image_path','image_sha256')
result={'scope':'BART only; ULMFiT ingestion failed; not the frozen two-paper benchmark','cases':{}}
for case in fixture['image_queries']:
    if case['id'] not in {'v3img02','v3img03'}: continue
    start=time.monotonic()
    row={'failure':None,'top5':None}
    try:
        found=visual.search_image(case['query_file'],top_k=5)
        row['top5']=[{k:h[k] for k in fields if k in h} for h in found['hits']]
    except Exception as exc:
        row['failure']=type(exc).__name__
    row['elapsed_seconds']=time.monotonic()-start
    result['cases'][case['id']]=row
target=Path('runs/multimodal-holdout-v3-bart-images-20261101/images.json')
target.parent.mkdir(parents=True,exist_ok=False)
with target.open('x',encoding='utf8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print('Recorded',len(result['cases']),'BART-only image searches')
