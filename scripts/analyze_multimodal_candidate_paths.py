"""Inspect recorded diagnostic candidate ranks without running retrieval."""
import json
from pathlib import Path

root = Path('runs/multimodal-holdout-v1-eval-20261031/traces')
targets = {'h01':'v-1f76660bf528d2d968a0dd93',
           'h04':'v-f445fad1b5b9b283dfc8a996',
           'h07':'v-f7ae3b58ddd9328d21034749'}
result = {}
for case, target in targets.items():
    trace = json.loads(next((root/('holdout-'+case)).rglob('holdout-'+case+'.json')).read_text(encoding='utf8'))
    stages = []
    for stage in trace['stages']:
        if stage['stage'] not in ('Retrieve','Rerank'):
            continue
        output = stage.get('output') or {}
        ranks = {}
        for key, values in output.items():
            if not isinstance(values, list) or key == 'routes':
                continue
            rank = next((i for i,v in enumerate(values,1)
                         if (v.get('chunk_id') if isinstance(v,dict) else v) == target), None)
            ranks[key] = {'rank':rank,'recorded_count':len(values)}
        routes = []
        for route in output.get('routes',[]):
            refs = route.get('refs',[])
            rank = next((i for i,v in enumerate(refs,1) if v.get('chunk_id') == target), None)
            routes.append({'route':route.get('route'),'rank':rank,'recorded_count':len(refs)})
        stages.append({'stage':stage['stage'],'ranks':ranks,'routes':routes})
    result[case] = {'target':target,'stages':stages}
print(json.dumps(result, ensure_ascii=False, indent=2))
