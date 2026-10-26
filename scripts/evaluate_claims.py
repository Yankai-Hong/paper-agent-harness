"""Score actual pipeline claims against the exact final Rerank evidence snapshot."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from paper_agent.claim_eval import evaluate_claim, summarize_claims, PROMPT
from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.storage import read_jsonl, write_json, write_jsonl
from claim_benchmark import recording_reasoner


def run(root, workers=4):
    root = Path(root)
    if (root/'claim-review.jsonl').exists():
        raise FileExistsError('Claim results already exist; use a separate run directory to preserve the original evaluation')
    preds = read_jsonl(root/('predictions.jsonl' if (root/'predictions.jsonl').exists() else 'per_case.jsonl'))
    chunks = {r['chunk_id']:r for r in read_jsonl('indexes/bge/chunks.jsonl')}
    config = replace(APIConfig.from_env(), max_tokens=4096, thinking='disabled')
    def score(pred):
        sid = pred['session_id']
        identifier = pred.get('turn_id', pred.get('id'))
        claims = (pred.get('answer') or {}).get('claims', [])
        records = []
        error = pred.get('error')
        evidence = []
        if claims:
            try:
                trace = json.loads((root/'traces'/sid/(pred.get('trace_id', identifier)+'.json')).read_text(encoding='utf-8'))
                last = {s['stage']:s for s in trace['stages']}
                ids = last.get('Rerank', {}).get('output', {}).get('evidence_ids', [])
                evidence = [chunks[i] for i in ids if i in chunks]
            except Exception as exc:
                error = type(exc).__name__
        for i, claim in enumerate(claims):
            if error:
                refs = list({(c.get('paper_id'),c.get('chunk_id')):c for c in claim.get('citations',[])}.values())
                result = {'text':claim.get('text'), 'citations':[
                    {**c,'valid':False,'validation_unknown':True,'verdict':'unknown'} for c in refs],
                    'verdict':'unknown', 'error':str(error), 'calls':[]}
            else:
                reasoner, raw = recording_reasoner(config, f'claim-{identifier}-{i}')
                result = evaluate_claim(claim, evidence, reasoner)
                result['raw_outputs'] = raw
            records.append({**result, 'case_id':identifier, 'session_id':sid, 'claim_index':i})
        case = {'id':identifier, 'session_id':sid, 'pipeline_error':error, 'claims_count':len(claims), 'claims':records}
        write_json(root/'claim-review'/(identifier+'.json'), case)
        return case
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(score, preds))
    records = [r for c in cases for r in c['claims']]
    write_jsonl(root/'claim-review.jsonl', records)
    summary = summarize_claims(records)
    summary.update(cases_total=len(cases), pipeline_failures=sum(bool(c['pipeline_error']) for c in cases),
                   cases_without_claims=sum(not c['claims_count'] for c in cases), prompt=PROMPT,
                   model=config.model, human_gold_validation='unavailable; no human labels supplied',
                   coverage='Generate.claims only; free answer prose not independently segmented')
    write_json(root/'claim-metrics.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    print(run(args.run))
