"""Verify immutable retrieval dependencies before reusing an independent local run."""
import hashlib
import json
from pathlib import Path
import shutil
from paper_agent.storage import digest, read_jsonl, write_json

source = Path('.cache/audit-20261026-retrieval')
out = Path('reports/final-acceptance/retrieval-reused')
config = json.loads((source/'run_config.json').read_text())
checks = {}
for name in ['retrieval.py','corpus.py','evaluation.py','storage.py']:
    path = Path('src/paper_agent')/name
    old = next(v for k,v in config['source_hashes'].items() if k.replace('\\','/')==path.as_posix())
    checks[name] = hashlib.sha256(path.read_bytes()).hexdigest()==old
checks['encoder_provenance'] = json.loads(Path('models/bge-m3/provenance.json').read_text())==config['encoder_provenance']
checks['reranker_provenance'] = json.loads(Path('models/bge-reranker-v2-m3/provenance.json').read_text())==json.loads((source/'reranker_provenance.json').read_text())
index = json.loads(Path('indexes/bge/metadata.json').read_text())
checks['index_sha256'] = hashlib.sha256(Path('indexes/bge/dense.faiss').read_bytes()).hexdigest()==index['index_sha256']
for mode in ['bm25','dense','hybrid','rerank']:
    m = json.loads((source/mode/'metrics.json').read_text())
    checks[mode+'_corpus'] = m['corpus_hash']==digest(read_jsonl('indexes/bge/chunks.jsonl'))
    checks[mode+'_queries'] = m['queries_hash']==digest(read_jsonl('data/qasper/retrieval.jsonl'))
    assert len(read_jsonl(source/mode/'predictions.jsonl'))==100
assert all(checks.values()), checks
out.mkdir(parents=True,exist_ok=False)
for path in source.rglob('*'):
    if path.is_file() and path.suffix in {'.json','.jsonl'}:
        target = out/path.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
write_json(out/'reuse-verification.json',{'reused_from':str(source),'checks':checks,
    'not_a_new_run':True,'reason':'All retrieval dependencies and predictions verified; memory changes do not alter standalone retrieval.',
    'scope':'100 known-paper queries; global corpus is a different task, not measured by these numbers'})
print(checks)
