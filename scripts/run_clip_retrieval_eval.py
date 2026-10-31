"""Run fixed text-to-figure and image-to-figure retrieval without an LLM.

The fixed gold locations are assistant-inspected, not independently reviewed.
The full actual rankings are recorded; failed evidence association remains unknown.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever, _number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('visual-python', 'image-model', 'text-model', 'visual-index', 'image-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite previous results')
    case_path = Path('data/eval/multimodal-v1/cases.json')
    raw = case_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'c74a7cb62105e09165fea77d2d86fc08dbe09a34fd8b991728d21ebe85f02fbf':
        raise ValueError('Fixed evaluation fixture hash mismatch')
    cases = json.loads(raw)
    encoder = SubprocessEncoder(args.visual_python, args.image_model, args.text_model, args.device)
    retriever = VisualRetriever(args.visual_index, args.image_root, encoder)
    text_cases = [case for case in cases['cases'] if case['id'] in {'a01','a02','a03','c01','c02','m01'}]
    start = time.perf_counter()
    text_vectors = encoder.encode_texts([case['question'] for case in text_cases])
    text_seconds = time.perf_counter() - start
    start = time.perf_counter()
    image_vectors = encoder.encode_images([case['query_file'] for case in cases['image_queries']])
    image_seconds = time.perf_counter() - start
    result = {'fixture_sha256': hashlib.sha256(raw).hexdigest(), 'model_provenance': encoder.provenance,
              'index_metadata': retriever.metadata, 'device': args.device,
              'batch_encoding_seconds': {'texts': text_seconds, 'images': image_seconds}, 'cases': []}
    fields = ('paper_id', 'chunk_id', 'evidence_type', 'page', 'bbox', 'figure_number', 'image_path', 'rank', 'visual_score')
    def evaluate(case_id, query_type, vector, paper_id, page, figure):
        start = time.perf_counter()
        hits = retriever._search(vector.reshape(1, -1), 5, None, None, start)['hits']
        expected = [e for e in retriever.evidence if e['paper_id'] == paper_id and e.get('page') == page
                    and _number(e.get('figure_number')) == _number(figure)]
        expected_keys = {(e['paper_id'], e['chunk_id']) for e in expected}
        row = {'id': case_id, 'query_type': query_type, 'target': {'paper_id': paper_id, 'page': page, 'figure_number': figure},
               'candidate_target_count': len(expected),
               'hit_at_5': any((e['paper_id'], e['chunk_id']) in expected_keys for e in hits) if len(expected) == 1 else None,
               'top5': [{k: e[k] for k in fields if k in e} for e in hits],
               'search_seconds': time.perf_counter() - start}
        # Verify both constraint types on actual index results, not substituted rankings.
        scoped = retriever._search(vector.reshape(1, -1), 5, [paper_id], figure, time.perf_counter())['hits']
        row['filtered_top5'] = [{k: e[k] for k in fields if k in e} for e in scoped]
        row['scope_filter_valid'] = (bool(scoped) and all(e['paper_id'] == paper_id and _number(e.get('figure_number')) == _number(figure) for e in scoped)) if len(expected) == 1 else None
        result['cases'].append(row)
    for case, vector in zip(text_cases, text_vectors):
        ev = next(e for e in case['evidence'] if e['locator'].startswith(('Figure', 'Table')))
        evaluate(case['id'], 'text', vector, case['paper_id'], ev['page'], ev['locator'])
    for case, vector in zip(cases['image_queries'], image_vectors):
        evaluate(case['id'], 'image', vector, case['paper_id'], case['target_page'], case['target_locator'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'result_path': str(output), 'cases': len(result['cases']), 'device': args.device}, ensure_ascii=False))


if __name__ == '__main__':
    main()
