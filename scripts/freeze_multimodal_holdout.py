"""Record the fixture and implementation hashes before holdout execution."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

paths = sorted(Path('src/paper_agent').glob('*.py')) + [
    Path('scripts/docling_ingest.py'), Path('scripts/clip_worker.py'),
    Path('scripts/run_multimodal_acceptance.py'), Path('scripts/run_multimodal_holdout.py'),
    Path('data/eval/multimodal-holdout-v1/cases.json')]
manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
    'status':'implementation frozen before holdout parsing, retrieval, or model inference',
    'sha256':{str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
output = Path('reports/revision/MULTIMODAL-HOLDOUT-FREEZE-20261031.json')
with output.open('x', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(output)
