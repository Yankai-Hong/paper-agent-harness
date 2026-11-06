"""Summarize per-request local ledgers without reading model response text."""
import argparse
from collections import Counter
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('runs', nargs='+')
args = p.parse_args()
result = {}
for name in args.runs:
    paths = sorted((Path(name)/'requests').glob('*.json'))
    calls = [json.loads(path.read_text(encoding='utf8')) for path in paths]
    complete = [r for r in calls if r.get('usage_complete')]
    result[name] = {'calls':len(calls), 'status':dict(Counter(r.get('status') for r in calls)),
        'stages':dict(Counter(r.get('stage') for r in calls)), 'usage_complete':len(complete),
        'prompt_tokens_known':sum(r.get('usage_raw',{}).get('prompt_tokens',0) for r in complete),
        'completion_tokens_known':sum(r.get('usage_raw',{}).get('completion_tokens',0) for r in complete),
        'total_tokens_known':sum(r['usage_raw']['prompt_tokens'] + r['usage_raw']['completion_tokens'] for r in complete),
        'elapsed_seconds':round(sum(r.get('elapsed_ms',0) for r in calls)/1000,2),
        'cost':'unknown'}
print(json.dumps(result,ensure_ascii=False,indent=2))
