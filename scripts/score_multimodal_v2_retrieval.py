"""Score recorded rankings against the independent v2 source-page mapping.

No retrieval, inference or semantic answer judgement is performed here. A group
contains alternative IDs for one necessary gold item; all groups are necessary
for strict full-evidence recall. A question never reaching retrieval is a miss
in the task-level denominator. See the index audit for the mapping and caveats.
"""
import json
from collections import defaultdict
from pathlib import Path

groups = {
    'v201':[['v-aa353350eacefb1f8c95708d','v-2fbf5a8801a0edf308bc1a98']],
    'v202':[['v-a95e7d5f2b33cf30e5636ebd'],['v-bcb662d63ac62b1b64676acc']],
    'v203':[['v-bd5fb1a3c155e453b94417a8']],
    'v204':[['v-7c7dfc257c4fed7da8703472']],
    'v205':[['v-2a91c8d9e30bafb52e70bf11'],['v-de9db05380e0050f74462976'],['v-9430f2cd13d4c5067c84ad78']],
    'v206':[['v-9d13e4571a7ff69ff1dd7cd0'],['v-6c0fc2e992a434c5407ea63a'],['v-69faa7d5902f8ccc38b011c3']],
    'v207':[['v-aaa51bde12f46c1511122763']],
    'v208':[['v-aa353350eacefb1f8c95708d'],['v-2a91c8d9e30bafb52e70bf11']],
    'v209':[['v-aa353350eacefb1f8c95708d']],
}
root = Path('runs/multimodal-holdout-v2-eval-20261031')
predictions = json.loads((root/'predictions.json').read_text(encoding='utf8'))['cases']
result = {'mapping_basis':'MULTIMODAL-HOLDOUT-V2-INDEX-AUDIT-20261031.md',
          'denominator':'v201-v209: 9 answerable questions; no retrieval is a task-level miss',
          'route_definition':'top five per recorded route and per subquery; any subquery may hit; not final fused ranking',
          'cases':{},'routes':{}}
route_cases = defaultdict(dict)
for case, gold in groups.items():
    row = predictions.get(case)
    if row is None:
        result['cases'][case] = {'execution':'not_executed'}
        continue
    top = [r['chunk_id'] for r in row.get('top5') or []]
    matching = [i for i,key in enumerate(top,1) if any(key in group for group in gold)]
    result['cases'][case] = {'retrieval_complete':row.get('retrieval_complete',False),
        'target_rank':min(matching) if matching else None,
        'all_gold_in_top5':all(any(key in group for key in top) for group in gold),
        'per_gold_group_hit':[any(key in group for key in top) for group in gold]}
    for trace_path in (root/'traces'/('holdout-'+case)).rglob('holdout-'+case+'.json'):
        trace = json.loads(trace_path.read_text(encoding='utf8'))
        for stage in trace.get('stages',[]):
            for route in (stage.get('output') or {}).get('routes',[]):
                name = route['route']
                refs = [ref['chunk_id'] for ref in route.get('refs',[])[:5]]
                present = any(any(key in group for key in refs) for group in gold)
                route_cases[name][case] = route_cases[name].get(case,False) or present

executed = [r for r in result['cases'].values() if r.get('retrieval_complete')]
result['summary'] = {'eligible':len(groups), 'executed_retrieval':len(executed),
    'hit_at_5':sum(r.get('target_rank') is not None for r in result['cases'].values()),
    'full_gold_at_5':sum(r.get('all_gold_in_top5',False) for r in result['cases'].values()),
    'mrr_at_5':sum(1/r['target_rank'] for r in result['cases'].values() if r.get('target_rank'))/len(groups)}
for name, cases in sorted(route_cases.items()):
    result['routes'][name] = {'hits':sum(cases.values()), 'cases_with_route':len(cases),
                              'task_denominator':len(groups)}
out = root/'retrieval-scores.json'
with out.open('x', encoding='utf8') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(result['summary'],ensure_ascii=False))
