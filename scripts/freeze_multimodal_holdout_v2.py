"""Freeze the v2 holdout, code and fixed runtime settings before PDF ingest."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

fixture = Path('data/eval/multimodal-holdout-v2/cases.json')
fixture_sha = '662f643fb38e95299c230ae7ae640ab0a6c5e33bad7817d9feb2b765c1443de6'
if hashlib.sha256(fixture.read_bytes()).hexdigest() != fixture_sha:
    raise ValueError('Fixture differs from the independently reviewed version')
paths = sorted(Path('src/paper_agent').glob('*.py')) + [
    Path('scripts/docling_ingest.py'), Path('scripts/clip_worker.py'),
    Path('scripts/run_multimodal_acceptance.py'), Path('scripts/run_multimodal_v3_eval.py'),
    Path('scripts/prepare_multimodal_holdout_v2.py'), Path('scripts/freeze_multimodal_holdout_v2.py'), fixture,
    Path('models/multimodal-v1/clip-image-v1/provenance.json'),
    Path('models/multimodal-v1/clip-text-v1/provenance.json'),
    Path('models/bge-m3/provenance.json'), Path('models/bge-reranker-v2-m3/provenance.json'),
    Path('data/raw/multimodal-eval/docling-models-manifest-20261030.json'),
    Path('reports/revision/MULTIMODAL-HOLDOUT-V2-PRE-AUDIT-20261031.md')]
manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
    'status':'frozen before this holdout was ingested, retrieved, or sent to a model',
    'settings':{'model':'deepseek-v4.1-flash', 'mode':'rerank', 'text_model':'models/bge-m3',
        'reranker':'models/bge-reranker-v2-m3', 'text_device':'cpu', 'visual_device':'cuda:0',
        'image_model':'models/multimodal-v1/clip-image-v1',
        'multilingual_text_model':'models/multimodal-v1/clip-text-v1',
        'visual_evidence_limit':3, 'final_retrieval_top_k':5, 'fresh_session_per_case':True,
        'decomposition':True, 'numeric_citation_repair_max':1,
        'question':'exact frozen question, no oracle scope prefix', 'automatic_retries':0,
        'max_output_tokens':1024, 'timeout_seconds':60, 'telemetry':'disabled'},
    'sha256':{str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
output = Path('reports/revision/MULTIMODAL-HOLDOUT-V2-FREEZE-20261031.json')
with output.open('x', encoding='utf8') as stream:
    json.dump(manifest, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(output)
