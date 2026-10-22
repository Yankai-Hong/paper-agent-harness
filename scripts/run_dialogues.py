"""Explicit API batch execution, session-isolated and traceable, no self-grading."""
import argparse
from pathlib import Path
import hashlib
from dataclasses import replace

from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.pipeline import PaperPipeline, PaperAgentState
from paper_agent.retrieval import Retriever, BGEEncoder, BGEReranker
from paper_agent.storage import read_jsonl, write_jsonl, write_json, digest, safe_id


def run(rows, retriever, config, output, mode='rerank', rewrite=True, reflect=True, compact=True, memory_strategy='bounded'):
    output = Path(output)
    if output.exists():
        raise ValueError('Output directory must be new, to prevent mixing experiments')
    seen, turns = set(), {}
    for row in rows:
        sid = safe_id(row['session_id'])
        if row['turn_id'] in seen or row['turn'] != turns.get(sid, 0)+1:
            raise ValueError('Need unique turn IDs and ordered complete session prefixes')
        seen.add(row['turn_id'])
        turns[sid] = row['turn']
    if not rows:
        raise ValueError('Empty dialogue file')
    write_json(output/'run_config.json', {'dataset_hash': digest(rows), 'corpus_hash': retriever.metadata['corpus_hash'],
        'model': config.model, 'base_url': config.base_url, 'mode': mode,
        'rewrite': rewrite, 'reflect': reflect, 'compact': compact, 'memory_strategy': memory_strategy, 'max_calls': config.max_calls,
        'max_tokens': config.max_tokens, 'json_mode': config.json_mode, 'thinking': config.thinking,
        'dataset_review_statuses': sorted({r.get('review_status', 'unspecified') for r in rows})})
    source_root = Path(__file__).resolve().parents[1]
    write_json(output/'source_hashes.json', {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src/paper_agent', 'scripts') for p in sorted((source_root/folder).glob('*.py'))})
    states, reasoners, results = {}, {}, []
    calls = 0
    for row in rows:
        sid = row['session_id']
        state = states.setdefault(sid, PaperAgentState(sid))
        reasoner = reasoners.setdefault(sid, CompatibleReasoner(config, sid))
        # Shared batch cap across sessions, including failed requests.
        reasoner.config = replace(config, max_calls=max(1, config.max_calls-calls+len(reasoner.calls)))
        before = len(reasoner.calls)
        pipeline = PaperPipeline(retriever, reasoner, output/'traces'/sid, output/'memory',
                                 mode, rewrite, reflect, compact, memory_strategy)
        try:
            if calls >= config.max_calls:
                raise RuntimeError('Batch API_MAX_CALLS reached')
            result = pipeline.run(row['question'], state)
            results.append({'turn_id': row['turn_id'], 'session_id': sid, **result})
        except Exception as exc:
            results.append({'turn_id': row['turn_id'], 'session_id': sid, 'error': str(exc)})
        calls += len(reasoner.calls)-before
        write_jsonl(output/'predictions.jsonl', results)
        print(f"Turn {row['turn_id']}: {'error' if 'error' in results[-1] else results[-1]['answer']['decision']}; API attempts {calls}", flush=True)
    usage_rows = [call for reasoner in reasoners.values() for call in reasoner.calls]
    write_json(output/'usage.json', {'calls': calls, 'all_usage_complete': all(c.get('usage_complete', False) for c in usage_rows),
        'sessions': {sid: {key: sum(c.get('usage', {}).get(key, 0) for c in reasoner.calls)
            for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')}
            for sid, reasoner in reasoners.items()}, 'call_records': usage_rows})
    print(f'Saved {len(results)} turns, {calls} API attempts. No semantic correctness scores inferred.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dialogues')
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', default='indexes/bge')
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--mode', choices=['bm25', 'dense', 'hybrid', 'rerank'], default='rerank')
    parser.add_argument('--encoder', default='models/bge-m3')
    parser.add_argument('--reranker', default='models/bge-reranker-v2-m3')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-rewrite', action='store_true')
    parser.add_argument('--no-reflect', action='store_true')
    parser.add_argument('--no-compact', action='store_true')
    parser.add_argument('--memory-strategy', choices=['bounded','semantic'], default='bounded')
    args = parser.parse_args()
    config = APIConfig.from_env(args.env_file)
    if Path(args.output).exists():
        parser.error('Output directory must be new')
    encoder = BGEEncoder(args.encoder, args.device, 4) if args.mode != 'bm25' else None
    reranker = BGEReranker(args.reranker, args.device, 4) if args.mode == 'rerank' else None
    run(read_jsonl(args.dialogues), Retriever(args.index, encoder, reranker), config, args.output,
        args.mode, not args.no_rewrite, not args.no_reflect, not args.no_compact, args.memory_strategy)
