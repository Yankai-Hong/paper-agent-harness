"""Small paired QASPER evaluation with actual shared prior turns, no Judge.

--dry-run never reads credentials or makes a network request. The user must
choose a new output directory; old reports and datasets are never overwritten.
"""
import argparse
import copy
import hashlib
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

from paper_agent.evaluation import evidence_units
from paper_agent.memory_manager import MemoryManager
from paper_agent.observability import Exporter, local_trace_path
from paper_agent.pipeline import PaperAgentState, PaperPipeline
from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.retrieval import Retriever
from paper_agent.storage import digest, read_jsonl, write_json


MAX_REQUESTS = 36
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_CHARS = 30000
REQUEST_TIMEOUT = 45
TURN_TIMEOUT = 140


def fixed_cases(rows):
    papers = []
    for row in rows:
        if row['source_paper_id'] not in {item['source_paper_id'] for item in papers}:
            papers.append(row)
        if len(papers) == 3:
            break
    if len(papers) != 3:
        raise ValueError('Three distinct fixed QASPER papers are required')
    a, b, c = papers
    aid, bid, cid = (item['source_paper_id'] for item in papers)
    seeds = [
        {'id': 'seed-earlier', 'question': f"For paper {cid}: {c['query']}", 'paper_id': cid},
        {'id': 'seed-latest', 'question': f"For paper {bid}: {b['query']}", 'paper_id': bid},
    ]
    cases = [
        {'id': 'simple', 'question': f"For paper {aid}: {a['query']}",
         'targets': [a], 'gold_targets': [a], 'needs_history': False},
        {'id': 'comparison', 'question': f"For paper {aid}: {a['query']} For paper {bid}: {b['query']}",
         'targets': [a, b], 'gold_targets': [a, b], 'needs_history': False},
        {'id': 'time-exclusion', 'question': f'比较论文 {aid} 和 {bid} 的研究方法，不包括2024年的数据。',
         'targets': [a, b], 'gold_targets': [], 'needs_history': False},
        {'id': 'one-side-missing', 'question': f"For paper {aid}: {a['query']} For paper {bid}, what is the exact SHA-256 hexadecimal checksum of its original training dataset?",
         'targets': [a, b], 'gold_targets': [a], 'needs_history': False},
        {'id': 'recent', 'question': f'比较论文 {aid} 和刚才讨论的那篇论文的研究方法。',
         'targets': [a, b], 'gold_targets': [], 'needs_history': True},
        {'id': 'singular', 'question': f'比较论文 {aid} 和它（上一轮讨论的论文）的研究方法。',
         'targets': [a, b], 'gold_targets': [], 'needs_history': True},
    ]
    return seeds, cases


def seed_sqlite_copy(manager, state):
    """Restore a prior actual session to a fresh, isolated SQLite namespace."""
    clone = copy.deepcopy(state)
    clone.memory_namespace = manager.identity
    snapshot = asdict(clone)
    with manager.store.transaction() as db:
        if db.execute('SELECT 1 FROM sessions WHERE session_id=?', (clone.session_id,)).fetchone():
            raise ValueError('Refuse to overwrite an existing session')
        db.execute('INSERT INTO sessions VALUES (?,?,?)',
                   (clone.session_id, clone.turn_count, json.dumps(snapshot, ensure_ascii=False)))
    restored = manager.store.snapshot(clone.session_id)
    if restored['messages'] != snapshot['messages'] or restored['committed_turns'] != snapshot['committed_turns']:
        raise RuntimeError('SQLite history copy failed verification')
    return digest({'messages': snapshot['messages'], 'ledger': snapshot['committed_turns']})


def protocol(seeds, cases):
    return {
        'case_ids': [case['id'] for case in cases], 'case_count': len(cases),
        'paired_conditions': ['baseline', 'split'], 'seed_turns_max': len(seeds),
        'max_actual_requests_total': MAX_REQUESTS, 'max_actual_requests_per_turn': 3,
        'max_output_tokens_per_request': MAX_OUTPUT_TOKENS,
        'max_possible_output_tokens': MAX_REQUESTS * MAX_OUTPUT_TOKENS,
        'max_input_chars_per_request': MAX_INPUT_CHARS,
        'per_request_timeout_seconds': REQUEST_TIMEOUT, 'per_turn_seconds': TURN_TIMEOUT,
        'transport_retries': 0, 'schema_repairs': 0, 'judge_requests': 0,
        'retrieval_rounds_max': 1, 'thinking': 'disabled', 'telemetry': 'disabled',
        'history': 'Two actual shared seed turns, cloned into isolated SQLite namespaces for each comparison.',
        'scope_labels': 'Author-created fixed IDs; not human semantic quality labels.',
        'quality_labels': 'Answer completeness, semantic support and false refusal are unknown.',
        'time_constraint_metric': '2024 literal in NLU rewrite is only a surface proxy; it does not validate negation.',
        'coverage_metric': 'Paper presence and official paragraph hits are retrieval proxies, not semantic support.',
        'data': 'Public QASPER v0.6 and existing BM25 index only; no private memory.',
        'dataset_hash': digest({'seed': seeds, 'cases': cases}),
    }


def summarize_attempt(item, trace, reasoner, before, started, retriever, case=None):
    calls = reasoner.calls[before:]
    item.update(elapsed_seconds=time.perf_counter()-started, actual_requests=len(calls),
        calls=[{k: call.get(k) for k in ('stage', 'status', 'error_type', 'error_category',
               'finish_reason', 'elapsed_ms', 'usage_complete', 'usage', 'model_requested', 'model_returned')}
               for call in calls], commit_status=trace.get('commit_status', 'unknown'),
        retrieval_calls=trace.get('budget', {}).get('used', {}).get('subqueries') if trace else None)
    stages = trace.get('stages', [])
    nlu = next((stage for stage in stages if stage['stage'] == 'NLU' and stage.get('status') == 'ok'), None)
    nlu_output = nlu.get('output', {}) if nlu else {}
    item.update(nlu_paper_ids=nlu_output.get('paper_ids'),
                nlu_needs_clarification=nlu_output.get('needs_clarification') if nlu else None,
                scope_guard=nlu.get('scope_guard') if nlu else None)
    if case:
        expected = [row['source_paper_id'] for row in case['targets']]
        observed = nlu_output.get('paper_ids')
        item['expected_paper_ids'] = expected
        item['target_resolution_match'] = (set(observed) == set(expected)) if isinstance(observed, list) else None
        item['time_2024_preserved_in_nlu'] = ('2024' in nlu_output.get('standalone_query', '')) if case['id'] == 'time-exclusion' and nlu else None
        evidence_ids = set()
        for stage in stages:
            if stage['stage'] == 'Rerank':
                evidence_ids.update(stage.get('output', {}).get('evidence_ids', []))
        evidence = [retriever.by_id[c] for c in evidence_ids if c in retriever.by_id]
        units = evidence_units(evidence)
        item['paper_coverage'] = {paper: any(e['paper_id'] == paper for e in evidence) if trace else None for paper in expected}
        item['official_gold_paragraph_hit'] = {
            row['query_id']: bool(units & set().union(*map(set, row.get('gold_alternatives') or [row['gold_units']]))) if trace else None
            for row in case['gold_targets']}
        item.update(answer_completeness=None, unsupported_conclusions=None, false_refusal=None,
                    quality_status='unknown; no independent semantic labels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Output must be a new directory')
    rows = read_jsonl('data/eval/v0.6/retrieval.jsonl')
    seeds, cases = fixed_cases(rows)
    spec = protocol(seeds, cases)
    if args.dry_run:
        print(json.dumps(spec, ensure_ascii=False))
        return
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', spec)
    summary = {'budget': spec, 'actual_requests_total': 0, 'seed_turns': [], 'runs': []}
    write_json(out/'summary.json', summary)
    retriever = Retriever('indexes/bge')
    missing_paper = next(case for case in cases if case['id'] == 'one-side-missing')['targets'][1]['source_paper_id']
    corpus = ' '.join(c['text'] for c in retriever.chunks if c['paper_id'] == missing_paper)
    hex_count = len(re.findall(r'(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', corpus))
    summary['missing_probe_64_hex_literals'] = hex_count
    summary['corpus_hash'] = retriever.metadata['corpus_hash']
    write_json(out/'summary.json', summary)
    if hex_count:
        raise ValueError('Fixed missing-evidence probe invalidated; no requests sent')
    paths = sorted(Path('src/paper_agent').glob('*.py')) + [Path(__file__), Path('data/eval/v0.6/retrieval.jsonl'), Path('indexes/bge/metadata.json')]
    write_json(out/'freeze.json', {'sha256': {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    config = replace(APIConfig.from_env(), max_calls=MAX_REQUESTS, max_retries=0, repair_retries=0,
                     max_tokens=MAX_OUTPUT_TOKENS, max_input_chars=MAX_INPUT_CHARS,
                     timeout=REQUEST_TIMEOUT, connect_timeout=10, thinking='disabled')
    reasoner = CompatibleReasoner(config, 'revision-multiturn')

    def pipeline(folder, enabled):
        memory = MemoryManager(folder/'memory', None, None, share_project=False)
        return PaperPipeline(retriever, reasoner, folder/'traces', folder/'memory', mode='bm25',
                             memory_manager=memory, decomposition=enabled, exporter=Exporter(enabled=False),
                             max_retrieval_attempts=1, budget_limits={'seconds':TURN_TIMEOUT, 'model_requests':3})

    def attempt(p, state, question, turn, run_id, case_id, case=None):
        item = {'case_id':case_id, 'condition':run_id, 'decision':None, 'error_type':None}
        before, started = len(reasoner.calls), time.perf_counter()
        if before >= MAX_REQUESTS:
            item.update(status='not_attempted_budget_exhausted', actual_requests=0,
                        target_resolution_match=None, answer_completeness=None,
                        unsupported_conclusions=None, false_refusal=None)
            return item
        try:
            result=p.run(question, state, turn_id=turn, case_id=case_id, run_id=run_id)
            item.update(decision=result['answer']['decision'], status='completed')
        except Exception as exc:
            item.update(error_type=type(exc).__name__, status='failed')
        path=local_trace_path(p.trace_dir, {'session_id':state.session_id, 'turn_id':turn, 'run_id':run_id})
        trace=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        item['trace_path']=str(path) if path.exists() else None
        summarize_attempt(item, trace, reasoner, before, started, retriever, case)
        return item

    seed_pipeline = pipeline(out/'seed', False)
    seed_state = PaperAgentState('shared-history')
    for seed in seeds:
        item = attempt(seed_pipeline, seed_state, seed['question'], seed['id'], 'seed', seed['id'])
        summary['seed_turns'].append(item)
        summary['actual_requests_total'] = len(reasoner.calls)
        write_json(out/'summary.json', summary)
    history_ready = (len(summary['seed_turns']) == 2 and all(
        item['status']=='completed' and item['commit_status']=='committed' and item['decision']=='answer'
        and item.get('nlu_paper_ids') == [seed['paper_id']]
        for item,seed in zip(summary['seed_turns'],seeds)))
    summary['shared_history_ready'] = history_ready
    summary['shared_history_hash'] = digest({'messages':seed_state.messages, 'ledger':seed_state.committed_turns}) if history_ready else None
    write_json(out/'summary.json', summary)
    for case in cases:
        for enabled in (False, True):
            condition = 'split' if enabled else 'baseline'
            folder = out/condition/case['id']
            p = pipeline(folder, enabled)
            if case['needs_history'] and not history_ready:
                item = {'case_id':case['id'], 'condition':condition, 'status':'not_attempted_history_unavailable',
                        'actual_requests':0, 'target_resolution_match':None, 'answer_completeness':None,
                        'unsupported_conclusions':None, 'false_refusal':None}
            else:
                state = PaperAgentState('shared-history' if case['needs_history'] else case['id'])
                history_hash = seed_sqlite_copy(p.memory_manager, seed_state) if case['needs_history'] else None
                item = attempt(p, state, case['question'], 'target', condition, case['id'], case)
                item['shared_history_hash'] = history_hash
            summary['runs'].append(item)
            summary['actual_requests_total'] = len(reasoner.calls)
            write_json(out/'summary.json', summary)
    print(json.dumps({'output':str(out/'summary.json'), 'actual_requests_total':len(reasoner.calls)}))


if __name__ == '__main__':
    main()
