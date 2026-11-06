"""Summarize actual local runs; gold IDs are for post-run scoring only."""
import argparse
import json
from pathlib import Path

GOLD = {
    'v301': [['v-048bf54e3c62918316858d40'], ['v-634d07fc63ec7383ca4980cb']],
    'v302': [['v-a4e9cf505dc170fd9903bd70'], ['v-6eb2eb027e9afb995f1da312']],
    'v303': [['v-7e971512098372bc26f89a4c']],
    'v304': [['v-f023738b85a3e064777c0d81'], ['v-959951e0c42d4a564d285ed4']],
    'v305': [['v-f545f55690e131b015c8a0ae'], ['v-048bf54e3c62918316858d40']],
    'v308': [['v-8b5eae33d2488c0053126e62'], ['v-1ddb443c2b36c90376d92bc3']],
    'v309': [['v-048bf54e3c62918316858d40']],
    'v3img01': [['v-048bf54e3c62918316858d40']],
    'v3img02': [['v-f545f55690e131b015c8a0ae']],
    'v3img03': [['v-a4e9cf505dc170fd9903bd70']],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('output')
    args = p.parse_args()
    root = Path(args.run)
    predictions = json.loads((root/'predictions.json').read_text(encoding='utf8'))['cases']
    result = {'scope':'two-paper development retest; not independent holdout', 'cases':{}}
    for cid in ['v301','v302','v303','v304','v305','v306','v307','v308','v309',
                'v3img01','v3img02','v3img03']:
        row = predictions.get(cid)
        if row is None:
            result['cases'][cid] = {'status':'not_executed'}
            continue
        ids = [x['chunk_id'] for x in (row.get('top5') or [])]
        groups = GOLD.get(cid)
        entry = {'status':'error' if row.get('failure') else 'executed',
                 'failure':row.get('failure'), 'decision':(row.get('answer') or {}).get('decision'),
                 'top5':row.get('top5'), 'request_ids':row.get('request_ids', []),
                 'target_ranks':[[ids.index(g)+1 for g in group if g in ids] for group in groups] if groups else None,
                 'any_target_top5':any(any(g in ids for g in group) for group in groups) if groups else None,
                 'all_targets_top5':all(any(g in ids for g in group) for group in groups) if groups else None}
        if cid.startswith('v3img'):
            for key in ('paper_oracle_top5','figure_oracle_top5','joint_oracle_top5'):
                entry[key]=row.get(key)
        else:
            traces=list((root/'traces'/('holdout-'+cid)).rglob('holdout-'+cid+'.json'))
            if len(traces) == 1:
                trace=json.loads(traces[0].read_text(encoding='utf8'))
                entry['budget_exhausted']=trace.get('budget_exhausted')
                entry['routes']=[{'stage':s['stage'], 'attempt':s.get('attempt'),
                    'route':route['route'], 'top5':route.get('refs', [])[:5]}
                    for s in trace.get('stages', []) for route in (s.get('output') or {}).get('routes', [])]
                entry['evidence_ids']=[s.get('output',{}).get('evidence_ids',[]) for s in trace.get('stages',[])
                                       if s.get('stage') == 'Rerank' and s.get('status') == 'ok']
            entry['claims']=[{'key':c.get('key'), 'text':c.get('text'), 'citations':c.get('citations',[])}
                             for c in (row.get('answer') or {}).get('claims', [])]
            entry['semantic_review']='requires agent original-page review'
        result['cases'][cid]=entry
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    main()
