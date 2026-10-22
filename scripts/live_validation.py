"""Run authorized live evaluation, keeping keys in memory and preserving .env."""
import argparse
from dataclasses import replace
from io import StringIO
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from paper_agent.providers import APIConfig, CompatibleReasoner
from paper_agent.storage import write_json, read_jsonl


def configuration(profile):
    config = APIConfig.from_env()
    if profile == 'configured':
        return config
    # Explicitly requested fallback only. Ignore commented example endpoints below it.
    pending = {}
    for line in Path('.env').read_text(encoding='utf-8-sig').splitlines():
        if not line.lstrip().startswith('#'):
            continue
        text = line.lstrip()[1:].lstrip()
        if '=' not in text:
            continue
        for key, value in dotenv_values(stream=StringIO(text)).items():
            if key in {'API_KEY', 'API_BASE_URL', 'MODEL_ID'}:
                pending[key] = value
        if (pending.get('API_KEY') and pending.get('MODEL_ID') and
                urlsplit(pending.get('API_BASE_URL', '')).hostname == 'api.deepseek.com'):
            return replace(config, api_key=pending['API_KEY'], model=pending['MODEL_ID'],
                           base_url=pending['API_BASE_URL'].rstrip('/'))
    raise ValueError('No complete commented DeepSeek profile found; .env was not changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['configured', 'deepseek-commented'], default='configured')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sessions', type=int, default=5)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--thinking', choices=['enabled', 'disabled'])
    parser.add_argument('--mode', choices=['bm25', 'dense', 'hybrid', 'rerank'], default='rerank')
    parser.add_argument('--no-rewrite', action='store_true')
    parser.add_argument('--no-reflect', action='store_true')
    parser.add_argument('--no-compact', action='store_true')
    args = parser.parse_args()
    config = configuration(args.profile)
    if args.thinking:
        config = replace(config, thinking=args.thinking)
    if args.smoke:
        reasoner = CompatibleReasoner(config, 'live-smoke-20261022')
        result = {}
        try:
            result = {'ok': True, 'result': reasoner.structured('nlu', {
                'question': 'Find evidence about retrieval in Example.', 'memory': {},
                'recent_messages': [], 'paper_catalog': [{'paper_id': 'example', 'title': 'Example'}]})}
        except Exception as exc:
            result = {'ok': False, 'error': str(exc)}
        write_json(args.output, {**result, 'model': config.model, 'calls': reasoner.calls})
        print(result)
        return
    from run_dialogues import run
    from paper_agent.retrieval import BGEEncoder, BGEReranker, Retriever
    import torch
    torch.set_num_threads(4)
    rows = read_jsonl('data/dialogues-draft/dialogues.jsonl')
    ids = list(dict.fromkeys(row['session_id'] for row in rows))[args.offset:args.offset+args.sessions]
    rows = [r for r in rows if r['session_id'] in ids]
    encoder = BGEEncoder('models/bge-m3', 'cuda:0', 4) if args.mode != 'bm25' else None
    reranker = BGEReranker('models/bge-reranker-v2-m3', 'cuda:0', 4) if args.mode == 'rerank' else None
    retriever = Retriever('indexes/bge', encoder, reranker)
    if args.workers == 1:
        run(rows, retriever, config, args.output, args.mode,
            not args.no_rewrite, not args.no_reflect, not args.no_compact)
    else:
        from concurrent.futures import ThreadPoolExecutor
        from threading import RLock
        # At most four calls per turn (NLU, two Reflect, Generate).
        if len(rows)*4 > config.max_calls:
            raise ValueError('Parallel batch worst-case calls exceed API_MAX_CALLS; reduce --sessions')
        if not 1 <= args.workers <= 4:
            raise ValueError('workers must be 1..4')
        lock = RLock()
        class LockedReranker:
            def score(self, *a, **kw):
                with lock:
                    return reranker.score(*a, **kw)
        class LockedRetriever:
            metadata = retriever.metadata
            by_id = retriever.by_id
            def __init__(self):
                self.reranker = LockedReranker() if reranker else None
            def search(self, *a, **kw):
                with lock:
                    return retriever.search(*a, **kw)
        shared = LockedRetriever()
        output = Path(args.output)
        if output.exists():
            raise ValueError('Output must be new')
        output.mkdir(parents=True)
        def one(sid):
            session_rows = [r for r in rows if r['session_id'] == sid]
            run(session_rows, shared, replace(config, max_calls=len(session_rows)*4), output/sid,
                args.mode, not args.no_rewrite, not args.no_reflect, not args.no_compact)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(one, ids))


if __name__ == '__main__':
    main()
