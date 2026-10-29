"""Read-only inventory: never export database payloads or environment secrets."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
out = root / 'reports/revision'
out.mkdir(exist_ok=True)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
files = {}
for directory in ('src', 'configs', 'data', 'models', 'indexes'):
    for p in (root/directory).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            files[p.relative_to(root).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
databases = []
for directory in ('runs', 'reports'):
    for p in (root/directory).rglob('*.sqlite3'):
        if 'revision' in p.parts:
            continue
        item = {'path': p.relative_to(root).as_posix(), 'sha256': sha(p)}
        try:
            with sqlite3.connect(p.resolve().as_uri()+'?mode=ro', uri=True) as db:
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                item['counts'] = {t: db.execute('SELECT count(*) FROM '+t).fetchone()[0]
                                  for t in ('facts','sessions','episodes') if t in tables}
                if 'facts' in tables:
                    item['sources'] = list(db.execute('SELECT source_type,status,count(*) FROM facts GROUP BY source_type,status'))
        except sqlite3.Error as e:
            item['error_type'] = type(e).__name__
        databases.append(item)
result = {'head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
          'python': sys.version, 'files': files, 'databases': databases,
          'api_evaluation_request_budget': 0,
          'turn_budget': {'model_requests': 8, 'subqueries': 6, 'candidates': 90, 'rerank': 60, 'seconds': 180},
          'deadline_note': 'Cooperative deadline; native GPU/DNS calls cannot be forcibly interrupted.'}
(out/'baseline.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'files':len(files),'databases':len(databases),'python':sys.version}))
