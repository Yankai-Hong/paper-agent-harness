"""Prepare a synthetic one-case corpus and BM25 index; no model or telemetry calls."""
import argparse
from pathlib import Path

from paper_agent.corpus import Chunker
from paper_agent.retrieval import build_index
from paper_agent.storage import digest, write_json, write_jsonl


PAPER = {
    'paper_id': 'synthetic-paper',
    'title': 'Synthetic Retrieval Example',
    'paragraphs': [{'paragraph_id': 'p1', 'section': 'Results',
                    'text': 'In this synthetic example, method Amber achieved 82 percent accuracy on the tiny test set.'}],
}
ROWS = [{
    'session_id': 'synthetic-session', 'turn_id': 'synthetic-case', 'turn': 1,
    'question': 'What accuracy did method Amber achieve in synthetic-paper?',
    'success_conditions': {'decision': 'answer', 'require_semantic_correctness': True},
    'review_status': 'synthetic_fixture_not_human_accuracy',
    'annotation_note': 'Fixed synthetic smoke fixture; rule scores are not semantic quality labels.',
}]


def prepare(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    chunks = Chunker().split(PAPER)
    write_jsonl(root/'papers.jsonl', [PAPER])
    write_jsonl(root/'chunks.jsonl', chunks)
    write_jsonl(root/'cases.jsonl', ROWS)
    metadata = build_index(root/'chunks.jsonl', root/'index')
    manifest = {'schema':'paper-agent-langfuse-synthetic-fixture-v1',
                'cases':len(ROWS), 'dataset_hash':digest(ROWS),
                'corpus_hash':metadata['corpus_hash'],
                'model_calls':0, 'external_uploads':0,
                'note':'Execution and citation structure scores are rules, not semantic correctness.'}
    write_json(root/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory; existing paths are never overwritten')
    args = parser.parse_args()
    prepare(args.output)
    print('Synthetic fixture and local BM25 index prepared; no external calls.')


if __name__ == '__main__':
    main()
