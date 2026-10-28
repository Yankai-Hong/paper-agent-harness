"""Offline ranking/ablation diagnostics; source evidence coverage is a proxy only."""
import json
from pathlib import Path
from paper_agent.storage import read_jsonl, write_json
from paper_agent.evaluation import evidence_units, retrieval_metrics
from paper_agent.eval_tasks import bootstrap_ci
from paper_agent.eval_protocol import success_metrics


def main():
    root=Path('reports/history/development/phase-a/full')
    chunks=read_jsonl('indexes/bge/chunks.jsonl')
    by_id={c['chunk_id']:c for c in chunks}
    queries=read_jsonl('data/qasper/retrieval.jsonl')
    for q in queries:
        gold=set().union(*map(set,q['gold_alternatives']))
        q['relevant_chunk_count']=sum(bool(evidence_units([c]) & gold) for c in chunks)
    groups={q['query_id']:q['source_paper_id'] for q in queries}
    ranking={}
    for mode in ['bm25','dense','hybrid','rerank']:
        predictions=read_jsonl(root/'retrieval'/mode/'predictions.jsonl')
        values=retrieval_metrics(queries,predictions,10)
        ranking[mode]={'at10':values,'ci':{k:bootstrap_ci([r[k] for r in values['per_query']],
            [groups[r['query_id']] for r in values['per_query']]) for k in ['recall','mrr','ndcg','hit']}}
    original={mode:json.loads((root/'retrieval'/mode/'metrics.json').read_text()) for mode in ranking}
    per={mode:{r['query_id']:r for r in m['per_query']} for mode,m in original.items()}
    ranking['complementary']={a+'_hit_'+b+'_miss':[q['query_id'] for q in queries if per[a][q['query_id']]['hit'] and not per[b][q['query_id']]['hit']]
        for a,b in [('bm25','dense'),('dense','bm25')]}
    ranking['ranking_changes']={b+'_vs_'+a:[{'query_id':q['query_id'],'recall_delta':per[b][q['query_id']]['recall']-per[a][q['query_id']]['recall'],
        'mrr_delta':per[b][q['query_id']]['mrr']-per[a][q['query_id']]['mrr']} for q in queries]
        for a,b in [('dense','hybrid'),('hybrid','rerank')]}
    write_json(root/'retrieval/extended-statistics.json',ranking)
    dialogs={r['turn_id']:r for r in read_jsonl('data/eval/v0.3/reflect-subset.jsonl')}
    diagnostics={}
    for run in (root/'ablations').iterdir():
        if not run.is_dir(): continue
        traces=[json.loads(p.read_text(encoding='utf-8')) for p in (run/'traces').rglob('*.json')]
        second=[]
        compressed=recall_turns=recall_records=0
        for t in traces:
            reranks=[s for s in t['stages'] if s['stage']=='Rerank']
            if len(reranks)>1:
                row=dialogs.get(t['turn_id'])
                coverage=[]
                if row and row['turn']<=3:
                    gold=set(row['reference_units'])
                    for stage in [reranks[0],reranks[-1]]:
                        hits=[by_id[c] for c in stage['output']['evidence_ids']]
                        coverage.append(len(evidence_units(hits)&gold)/len(gold))
                second.append({'turn_id':t['turn_id'],'source_evidence_coverage_proxy':coverage,
                    'improved_proxy':coverage[1]>coverage[0] if coverage else None})
            compressed+=any(s.get('compaction',{}).get('compressed') for s in t['stages'])
            archive=t.get('archive_recall',{})
            # Preserve provider-specific recall structure; do not invent a precision gold.
            if archive:
                recall_turns+=1
                recall_records+=len(archive)
        diagnostics[run.name]={'turns':len(traces),'second_retrieval_count':len(second),
            'second_retrieval_rate':len(second)/len(traces) if traces else None,
            'second_retrieval_cases':second,'compressed_turns':compressed,
            'archive_trace_present_turns':recall_turns,'archive_records_used':recall_records,'archive_precision':None,
            'note':'Follow-up source-evidence coverage is a proxy, not independently annotated sufficiency; archive precision lacks gold.'}
    write_json(root/'ablations/trace-diagnostics.json',diagnostics)
    ids={r['turn_id'] for r in read_jsonl('data/dialogues-draft/e2e-view.jsonl')}
    selected=[r for r in read_jsonl(root/'e2e/predictions.jsonl') if r['turn_id'] in ids]
    assert len(selected)==len(ids)==70
    write_json(root/'e2e-70-view.json',{'source':'full 120-turn replay; scoring view only, not a separate API run',
        'metrics':success_metrics(selected),'turn_ids':sorted(ids)})
    print('Saved retrieval @10 CIs, complementary cases, trace diagnostics and 70-turn view')


if __name__=='__main__': main()
