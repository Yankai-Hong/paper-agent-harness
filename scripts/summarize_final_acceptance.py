"""Recompute final evidence tables from saved predictions, traces and actual usage."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from paper_agent.storage import read_jsonl, write_json, write_jsonl
from paper_agent.eval_protocol import success_metrics
from paper_agent.memory import validate_citations

ROOT = Path('reports/final-acceptance')
LIVE = ROOT/'live-v2'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def rate(n,d): return {'numerator':n,'denominator':d,'rate':n/d if d else None}
def percent(m):
    if m.get('rate') is None: return 'unknown'
    return f"{m.get('numerator',m.get('correct'))}/{m.get('denominator',m.get('total'))} ({m['rate']:.2%})"


def main():
    summary = {'status':'model_assisted_not_human_reviewed','configuration':read(LIVE/'freeze.json')}
    intents = read_jsonl('data/eval/v0.6/intent.jsonl')
    ip = {r['id']:r for r in read_jsonl(LIVE/'intent/per_case.jsonl')}
    summary['intent'] = {'metrics':read(LIVE/'intent/metrics.json'),'class_counts':dict(Counter(r['intent'] for r in intents)),
        'errors':[{'id':r['id'],'question':r['question'],'expected':r['intent'],'actual':ip.get(r['id']),
                   'label_status':r['review_status']} for r in intents if ip.get(r['id'],{}).get('intent')!=r['intent']],
        'label_status_counts':dict(Counter(r['review_status'] for r in intents))}
    rewrites = read_jsonl(LIVE/'rewrite/review.jsonl')
    summary['rewrite'] = {'n':len(rewrites),'metrics':read(LIVE/'rewrite/metrics.json'),
        'review':{k:rate(sum(bool((r.get('judgment') or {}).get(k)) for r in rewrites),len(rewrites)) for k in
                  ['reference_correct','meaning_constraints_preserved','independently_searchable']},
        'all_three':rate(sum(bool(r.get('judgment')) and all(r['judgment'][k] for k in
                    ['reference_correct','meaning_constraints_preserved','independently_searchable']) for r in rewrites),len(rewrites)),
        'unknown':sum(r.get('judgment') is None for r in rewrites),
        'applicability':'90 includes 10 ambiguous cases; independent retrievability can correctly be false. Same-model auxiliary agreement.'}
    applicable={r['id'] for r in read_jsonl('data/eval/v0.6/query_rewrite.jsonl') if r.get('entity_preservation_applicable',True)}
    selected=[r for r in rewrites if r['id'] in applicable]
    summary['rewrite']['paper_followup_subset']={
        k:rate(sum(bool((r.get('judgment') or {}).get(k)) for r in selected),len(selected))
        for k in ['reference_correct','meaning_constraints_preserved','independently_searchable']}
    summary['rewrite']['paper_followup_all_three']=rate(sum(bool(r.get('judgment')) and all(r['judgment'][k] for k in
        ['reference_correct','meaning_constraints_preserved','independently_searchable']) for r in selected),len(selected))
    summary['retrieval'] = read(ROOT/'retrieval-reused/summary.json')
    summary['retrieval_reuse'] = read(ROOT/'retrieval-reused/reuse-verification.json')
    boundary = read_jsonl('data/eval/v0.6/boundary.jsonl')
    bp = {r['id']:r for r in read_jsonl(LIVE/'boundary/per_case.jsonl')}
    summary['boundary'] = {'metrics':read(LIVE/'boundary/metrics.json'),
        'failures':[r for r in bp.values() if r.get('error')],
        'same_run_e2e_reused':sum(r.get('provenance') is not None for r in bp.values())}
    # Categories describe the input contract, not inferred semantic ground truth.
    categories = {}
    for row in boundary:
        category = 'ambiguity' if row['decision']=='clarify' else ('out_of_scope' if any(x in row['question'] for x in ['银行','天气','股票','私人']) else 'evidence_or_answerability')
        categories.setdefault(category,[]).append(row)
    summary['boundary']['categories'] = {k:{**rate(sum(bp.get(r['id'],{}).get('decision')==r['decision'] for r in rows),len(rows)),
                                                    'ids':[r['id'] for r in rows]} for k,rows in categories.items()}
    chunks = {r['chunk_id']:r for r in read_jsonl('indexes/bge/chunks.jsonl')}
    failures, claims, trace_stats, quality = [], [], {}, {}
    for name in ['e2e','memory-full','memory-bounded','reflect-off','reflect-on','reflect-second']:
        out = LIVE/name
        predictions = read_jsonl(out/'predictions.jsonl')
        reviews = {r['id']:r for r in read_jsonl(out/'review.jsonl')}
        stats = Counter()
        for pred in predictions:
            t = read(out/'traces'/pred['session_id']/(pred['turn_id']+'.json'))
            stats['turns']+=1
            stats['committed']+=t.get('commit_status')=='committed'
            stats['degraded']+=bool(t.get('degraded'))
            stats['recall_turns']+=bool(t.get('archive_recall'))
            stats['archive_record_turns']+=any(r.get('retrieval_source')=='archive' for r in t.get('archive_recall',[]))
            stats['rerank_turns']+=any(s['stage']=='Rerank' and s.get('input',{}).get('candidate_ids') and s['status']=='ok' for s in t['stages'])
            stats['second_retrieval']+=sum(s['stage']=='Retrieve' for s in t['stages'])>1
            stats['compressed_turns']+=any(s.get('compaction',{}).get('compressed') for s in t['stages'])
            stats['three_layer_turns']+=t.get('memory_backend')=='three_layer'
            review = reviews[pred['turn_id']]
            j = review.get('judgment')
            if pred.get('error') or not j or not j['task_correct']:
                failures.append({'run':name,'id':pred['turn_id'],'error':pred.get('error'),
                    'judgment':j,'review_error':review.get('error'),'severity':'unknown' if not j else 'task_failure'})
            if name=='e2e':
                ids = next((s.get('output',{}).get('evidence_ids',[]) for s in reversed(t['stages']) if s['stage']=='Rerank'),[])
                evidence = [chunks[i] for i in ids]
                by_index = {c['index']:c for c in j['claims']} if j else {}
                for i,c in enumerate((pred.get('answer') or {}).get('claims',[])):
                    cr = {'id':pred['turn_id'],'claim_index':i,'text':c['text'],'verdict':by_index.get(i,{}).get('verdict','unknown'),
                          'reason':by_index.get(i,{}).get('reason'),'citations':[]}
                    refs = {r['index']:r for r in by_index.get(i,{}).get('citations',[])}
                    for k,ref in enumerate(c['citations']):
                        try:
                            validate_citations([{'text':c['text'],'citations':[ref]}],evidence)
                            valid=True
                        except (ValueError,KeyError,TypeError): valid=False
                        cr['citations'].append({**ref,'valid':valid,'verdict':refs.get(k,{}).get('verdict','unknown')})
                    claims.append(cr)
        trace_stats[name] = dict(stats)
        quality[name] = {'task_correct':rate(sum(bool(r.get('judgment')) and r['judgment']['task_correct'] for r in reviews.values()),len(predictions)),
                         'unknown':sum(not r.get('judgment') for r in reviews.values()),
                         'status':'same_model_auxiliary_not_human_accuracy'}
    summary['traces'], summary['auxiliary_quality'] = trace_stats,quality
    preds = read_jsonl(LIVE/'e2e/predictions.jsonl')
    summary['e2e120'] = success_metrics(preds)
    ids = {r['turn_id'] for r in read_jsonl('data/dialogues-draft/e2e-view.jsonl')}
    subset = [p for p in preds if p['turn_id'] in ids]
    assert len(subset)==70 and len(preds)==120
    summary['e2e70'] = {'source':'fixed scoring view of full 120 executed turns','ids':sorted(ids),'metrics':success_metrics(subset)}
    refs = [c for r in claims for c in r['citations']]
    summary['citations'] = {'claims':len(claims),'verdict_counts':{k:sum(r['verdict']==k for r in claims) for k in ['supported','insufficient','contradicted','unknown']},
        'claim_support':rate(sum(r['verdict']=='supported' for r in claims),len(claims)),
        'id_source_validity':rate(sum(c['valid'] for c in refs),len(refs)),
        'semantic_citation_support':rate(sum(c['verdict']=='supported' for c in refs),len(refs)),
        'scope':'Generate.claims and attached citations, same-model review; no independent human labels'}
    write_jsonl(ROOT/'claims.jsonl',claims)

    # Concrete memory contracts: current and historical values, conflicts, original
    # facts, early preferences and unresolved event recall. Do not count paper IDs.
    memory_checks = []
    for name in ['memory-full','memory-bounded']:
        out = LIVE/name
        rows = read_jsonl(out/'data.jsonl')
        with sqlite3.connect(out/'memory/facts.sqlite3') as db:
            db.row_factory = sqlite3.Row
            for sid in sorted({r['session_id'] for r in rows}):
                case = [r for r in rows if r['session_id']==sid]
                contract=case[0]['contract']
                facts=[dict(f) for f in db.execute('SELECT * FROM facts WHERE session_id=?',(sid,))]
                user=[f for f in facts if f['source_type']=='user_explicit']
                current=next((f for f in user if f['status']=='active'),{})
                snap=json.loads(db.execute('SELECT snapshot FROM sessions WHERE session_id=?',(sid,)).fetchone()[0])
                trace=read(out/'traces'/sid/(case[-1]['turn_id']+'.json'))
                recalled=trace.get('archive_recall',[])
                content=json.dumps(recalled,ensure_ascii=False)
                checks={
                    'current_updated_value':current.get('text')==contract['new'],
                    'old_version_retained':any(f['text']==contract['old'] and f['status']=='superseded' for f in user),
                    'unresolved_conflict_retained':any(f['text']=='UNRESOLVED_CONFLICT' and f['status']=='conflict' for f in user),
                    'conflict_not_active':not any(f['text']=='UNRESOLVED_CONFLICT' and f['status']=='active' for f in user),
                    'explicit_update_has_replaces':bool(current.get('replaces')),
                    'early_value_recalled':contract['old'] in content,
                    'updated_value_recalled':contract['new'] in content,
                    'unresolved_event_recalled':any(r.get('event_type')=='follow_up' for r in recalled),
                    'conflict_recalled_non_authoritative':any(r.get('status')=='conflict' and 'UNRESOLVED_CONFLICT' in str(r.get('content')) and not r.get('is_current_fact') and not r.get('is_authoritative') for r in recalled),
                    'full_or_bounded_context':len(snap['messages'])==(18 if name=='memory-full' else 8)}
                memory_checks.append({'run':name,'session':sid,'checks':checks,'current':current,'facts':facts,
                                      'final_recall':recalled,'contract':contract})
    write_jsonl(ROOT/'memory-checks.jsonl',memory_checks)
    summary['memory']={name:{'checks':rate(sum(v for r in memory_checks if r['run']==name for v in r['checks'].values()),
                                         sum(len(r['checks']) for r in memory_checks if r['run']==name)),
        'failed':[{'session':r['session'],'checks':[k for k,v in r['checks'].items() if not v]} for r in memory_checks if r['run']==name and not all(r['checks'].values())],
        'scenes':3,'turns':27,'quality':quality[name],'trace':trace_stats[name]} for name in ['memory-full','memory-bounded']}

    reservations = read_jsonl(LIVE/'requests.jsonl')
    costs = {}
    for reservation in reservations:
        path=LIVE/'request-results'/f"{reservation['request']:05d}.json"
        result=read(path) if path.exists() else {}
        group='judge' if reservation['category']=='judge' else next((name for name in
            ['memory-full','memory-bounded','reflect-off','reflect-on','reflect-second','boundary','e2e','intent','rewrite']
            if reservation['session'].startswith(name+'-')),'other')
        entry=costs.setdefault(group,{'requests':0,'known_tokens':0,'unknown_usage_requests':0,'returned_models':[]})
        entry['requests']+=1
        usage=result.get('usage') or {}
        if isinstance(usage.get('prompt_tokens'),int) and isinstance(usage.get('completion_tokens'),int):
            entry['known_tokens']+=usage['prompt_tokens']+usage['completion_tokens']
        else: entry['unknown_usage_requests']+=1
        if result.get('model') and result['model'] not in entry['returned_models']: entry['returned_models'].append(result['model'])
    summary['costs']=costs
    original=ROOT/'live'
    initial_reservations=read_jsonl(original/'requests.jsonl')
    initial={'requests':len(initial_reservations),'known_tokens':0,'unknown_usage_requests':0,
             'status':'interrupted discovery run; contains reused NLU and superseded partial E2E'}
    for r in initial_reservations:
        path=original/'request-results'/f"{r['request']:05d}.json"
        usage=(read(path).get('usage') or {}) if path.exists() else {}
        if isinstance(usage.get('prompt_tokens'),int) and isinstance(usage.get('completion_tokens'),int):
            initial['known_tokens']+=usage['prompt_tokens']+usage['completion_tokens']
        else: initial['unknown_usage_requests']+=1
    summary['initial_run_costs']=initial
    diagnostic_root=ROOT/'reflect-diagnostic-v2'
    if (diagnostic_root/'summary.json').exists():
        summary['reflect_diagnostic']=read(diagnostic_root/'summary.json')
        diagnostic_cost={'requests':summary['reflect_diagnostic']['requests'],'known_tokens':0,'unknown_usage_requests':0}
        for path in (diagnostic_root/'requests').glob('*.json'):
            usage=read(path).get('usage') or {}
            if isinstance(usage.get('prompt_tokens'),int) and isinstance(usage.get('completion_tokens'),int):
                diagnostic_cost['known_tokens']+=usage['prompt_tokens']+usage['completion_tokens']
            else: diagnostic_cost['unknown_usage_requests']+=1
        summary['diagnostic_costs']=diagnostic_cost
    else:
        diagnostic_cost={'requests':0,'known_tokens':0,'unknown_usage_requests':0}
    summary['overall_costs']={k:initial[k]+sum(r[k] for r in costs.values())+diagnostic_cost[k]
                              for k in ['requests','known_tokens','unknown_usage_requests']}
    summary['overall_costs']['request_limit']=1800
    if (ROOT/'integration-v2/result.json').exists():
        summary['final_real_integration']=read(ROOT/'integration-v2/result.json')
    full,bounded=costs['memory-full'],costs['memory-bounded']
    summary['memory_token_comparison']={'full':full,'bounded':bounded,'known_token_delta':bounded['known_tokens']-full['known_tokens'],
        'savings_rate':1-bounded['known_tokens']/full['known_tokens'] if full['known_tokens'] and not full['unknown_usage_requests'] and not bounded['unknown_usage_requests'] else None,
        'quality_full':quality['memory-full'],'quality_bounded':quality['memory-bounded'],
        'note':'All actual NLU/Reflect/Generate/retry calls included. Bounded compaction is deterministic, summary API calls=0. Judge separate. Currency cost unverified.'}
    write_jsonl(ROOT/'failures.jsonl',failures)
    # Fixed normal IDs are independent of outcomes; issue examples form a separate,
    # deliberately enriched set. The full failure catalog is not silently sampled away.
    fixed=['d01-t1','d02-t2','d03-t3','d04-t4','d05-t1','d06-t2','d07-t3','d08-t4']
    queue=[]
    reviews={r['id']:r for r in read_jsonl(LIVE/'e2e/review.jsonl')}
    for identifier in fixed:
        queue.append({'selection':'pre_fixed_normal','run':'e2e','id':identifier,'review':reviews[identifier],'human_reviewed':False})
    priority = {('e2e',c['id']):{'run':'e2e','id':c['id'],'severity':c['verdict']} for c in claims if c['verdict'] in ['contradicted','insufficient']}
    for f in failures:
        priority.setdefault((f['run'],f['id']),f)
    ordered = sorted(priority.values(),key=lambda f:({'unknown':0,'contradicted':1,'insufficient':2}.get(f['severity'],3),f['run'],f['id']))
    for f in ordered[:12]:
        rr=read(LIVE/f['run']/'review'/(f['id']+'.json'))
        queue.append({'selection':'problem_enriched','run':f['run'],'id':f['id'],'review':rr,'human_reviewed':False})
    for r in rewrites:
        if len(queue)>=24: break
        if not r.get('judgment') or not all(r['judgment'][k] for k in ['reference_correct','meaning_constraints_preserved','independently_searchable']):
            queue.append({'selection':'rewrite_problem','run':'rewrite','id':r['id'],'review':r,'human_reviewed':False})
    for r in memory_checks:
        if len(queue)>=30: break
        queue.append({'selection':'memory_contract','run':r['run'],'id':r['session'],'review':r,'human_reviewed':False})
    write_jsonl(ROOT/'review-package.jsonl',queue)
    summary['review_package']={'cases':len(queue),'selection':dict(Counter(r['selection'] for r in queue)),
        'all_problem_cases':len(failures),'full_catalog':'failures.jsonl','not_population_accuracy':True,'human_reviewed':False}
    write_json(ROOT/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['intent','rewrite','boundary','citations','memory','costs']},ensure_ascii=False)[:16000])


if __name__=='__main__': main()
