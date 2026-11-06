"""Freeze implementation, fixture and source hashes before v3 PDF parsing."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

fixture = Path('data/eval/multimodal-holdout-v3/cases.json')
expected = '833d3ea10bea410672a2974e3e1de5e1788c859ca34c83adcab950f6a97d6a1d'
if hashlib.sha256(fixture.read_bytes()).hexdigest() != expected:
    raise ValueError('Frozen v3 fixture hash mismatch')
paths = sorted(Path('src/paper_agent').glob('*.py')) + [
    Path('scripts/docling_ingest.py'), Path('scripts/apply_reviewed_composites.py'),
    Path('scripts/clip_worker.py'), Path('scripts/run_multimodal_acceptance.py'),
    Path('scripts/run_multimodal_v3_eval.py'), Path('scripts/prepare_multimodal_holdout_v3.py'),
    Path('scripts/freeze_multimodal_holdout_v3.py'), fixture,
    Path('reports/revision/MULTIMODAL-HOLDOUT-V3-PRE-AUDIT-20261101.md'),
    Path('models/multimodal-v1/clip-image-v1/provenance.json'),
    Path('models/multimodal-v1/clip-text-v1/provenance.json'),
    Path('models/bge-m3/provenance.json'), Path('models/bge-reranker-v2-m3/provenance.json'),
    Path('data/raw/multimodal-eval/docling-models-manifest-20261030.json')]
manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
    'status':'frozen before new PDFs were parsed, indexed, retrieved or sent to model',
    'settings':{'model':'deepseek-v4.1-flash','mode':'rerank','decomposition':True,
        'encoder':'models/bge-m3','reranker':'models/bge-reranker-v2-m3',
        'text_device':'cpu','visual_device':'cuda:0','figure_model':'models/multimodal-v1/clip-image-v1',
        'text_clip_model':'models/multimodal-v1/clip-text-v1','final_top_k':5,
        'max_visual_images':3,'request_timeout_seconds':60,'max_output_tokens':1024,
        'automatic_retries':0,'numeric_citation_repair_max':1,'fresh_session_per_case':True,
        'question':'verbatim fixture, without oracle paper prefix','telemetry':'disabled'},
    'sha256':{str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
target = Path('reports/revision/MULTIMODAL-HOLDOUT-V3-FREEZE-20261101.json')
with target.open('x',encoding='utf8') as stream:
    json.dump(manifest,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(target)
