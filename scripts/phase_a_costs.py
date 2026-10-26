"""Count canonical stored request records once, including retries and failed runs."""
from pathlib import Path
from report_phase_a import load
from paper_agent.storage import read_jsonl, write_json


def main():
    root=Path('reports/phase-a')
    sources={}
    for path in root.rglob('api_calls.json'):
        sources[str(path)]=load(path)
    for path in root.rglob('*.json'):
        if 'traces' in path.parts:
            sources[str(path)]=load(path).get('api_calls',[])
    for filename in ['semantic-review.jsonl','claim-review.jsonl']:
        for path in root.rglob(filename):
            sources[str(path)]=[c for r in read_jsonl(path) for c in r.get('calls',[])]
    fixture=root/'claim-benchmark/cases.jsonl'
    if fixture.exists(): sources[str(fixture)]=[c for r in read_jsonl(fixture) for c in r.get('calls',[])]
    rows=[]
    for path,calls in sources.items():
        rows.append({'source':path,'calls':len(calls),'known_tokens':sum(sum(c.get('usage',{}).values()) for c in calls),
            'unknown_usage':sum(not c.get('usage_complete') for c in calls),
            'retries':sum(bool(c.get('retry_reason')) for c in calls),
            'failed_requests':sum(c.get('status')!='ok' for c in calls)})
    smoke=load(root/'smoke/smoke.json')['checks']
    smoke_tokens=sum((v.get('usage',{}).get('total_tokens') or sum(v.get('usage',{}).get(k,0) for k in
        ['input_tokens','output_tokens','cache_read_input_tokens','cache_creation_input_tokens'])) for v in smoke.values())
    write_json(root/'cost-ledger.json',{'sources':rows,'saved_calls':sum(r['calls'] for r in rows),
        'known_tokens':sum(r['known_tokens'] for r in rows),'unknown_usage_calls':sum(r['unknown_usage'] for r in rows),
        'retry_attempts':sum(r['retries'] for r in rows),'smoke_separately':{'calls':2,'known_tokens':smoke_tokens},
        'monetary_cost':None,'remaining_quota':None,
        'note':'Lower bound from saved actual usage. Includes initial/repaired reruns and auxiliary judges. Canonical trace/call/JSONL sources only; JSON copies and raw response copies are not double counted.'})
    print('Saved calls',sum(r['calls'] for r in rows),'known tokens',sum(r['known_tokens'] for r in rows))


if __name__=='__main__': main()
