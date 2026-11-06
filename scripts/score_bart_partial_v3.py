"""Score three recorded BART-only diagnostics against agent-checked sources.

The missing second paper means this is not the frozen two-paper benchmark.
No model call, reranking, or semantic evaluation occurs here.
"""
from collections import defaultdict
import json
from pathlib import Path

gold={'v302':[['v-a4e9cf505dc170fd9903bd70'],['v-6eb2eb027e9afb995f1da312']],
      'v303':[['v-7e971512098372bc26f89a4c']],
      'v308':[['v-8b5eae33d2488c0053126e62'],['v-1ddb443c2b36c90376d92bc3']]}
root=Path('runs/multimodal-holdout-v3-bart-partial-20261101')
pred=json.loads((root/'predictions.json').read_text(encoding='utf8'))['cases']
result={'scope':'BART only; not the two-paper holdout','cases':{},'routes':{}}
routes=defaultdict(dict)
for cid,groups in gold.items():
    row=pred[cid]
    ids=[h['chunk_id'] for h in row.get('top5') or []]
    ranks=[i for i,k in enumerate(ids,1) if any(k in group for group in groups)]
    result['cases'][cid]={'target_rank':min(ranks) if ranks else None,
        'all_gold_in_top5':all(any(k in group for k in ids) for group in groups),
        'failure':row.get('failure')}
    for path in (root/'traces'/('holdout-'+cid)).rglob('holdout-'+cid+'.json'):
        trace=json.loads(path.read_text(encoding='utf8'))
        for stage in trace['stages']:
            for route in (stage.get('output') or {}).get('routes',[]):
                ids=[ref['chunk_id'] for ref in route.get('refs',[])[:5]]
                hit=any(any(key in group for key in ids) for group in groups)
                routes[route['route']][cid]=routes[route['route']].get(cid,False) or hit
result['summary']={'questions':3,'target_at_5':sum(v['target_rank'] is not None for v in result['cases'].values()),
                   'all_gold_at_5':sum(v['all_gold_in_top5'] for v in result['cases'].values())}
for name,values in sorted(routes.items()):
    result['routes'][name]={'hits':sum(values.values()),'observed_cases':len(values),
                            'interpretation':'top five per recorded query/subquery; not final fused results'}
out=root/'retrieval-scores.json'
with out.open('x',encoding='utf8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps(result['summary'],ensure_ascii=False))
