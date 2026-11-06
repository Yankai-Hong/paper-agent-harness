"""Small two-paper development evaluation; targets never constrain candidates."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--clip-text-index', required=True)
    p.add_argument('--visual-index', required=True)
    p.add_argument('--visual-root', required=True)
    p.add_argument('--visual-python', required=True)
    p.add_argument('--image-model', required=True)
    p.add_argument('--text-model', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists(): raise ValueError('Choose a new output file')
    from paper_agent.clip_text_index import ClipTextRetriever
    from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever
    from paper_agent.retrieval import Retriever
    from paper_agent.pipeline import _expand_located_context
    enc = SubprocessEncoder(args.visual_python, args.image_model, args.text_model, args.device)
    visual = VisualRetriever(args.visual_index, args.visual_root, enc)
    text = ClipTextRetriever(args.clip_text_index, enc)
    bm25 = Retriever('indexes/multimodal-acceptance-repair-text-20261102')
    if text.metadata['source_hash'] != bm25.metadata['corpus_hash']:
        raise ValueError('Text corpus mismatch')
    fixture = json.loads(Path('data/eval/multimodal-holdout-v3/cases.json').read_text(encoding='utf-8'))
    cases = fixture['image_queries']
    expected_visual = ['v-048bf54e3c62918316858d40','v-f545f55690e131b015c8a0ae','v-a4e9cf505dc170fd9903bd70']
    expected_text = ['v-634d07fc63ec7383ca4980cb','v-504b9885e5c0a063225d0a72','v-1faf7ec1d547486b39cb79a1']
    for case in cases:
        if hashlib.sha256(Path(case['query_file']).read_bytes()).hexdigest() != case['query_sha256']:
            raise ValueError('Query image checksum mismatch')
    started = time.monotonic()
    image_vectors = enc.encode_images([x['query_file'] for x in cases])
    output = {'scope':'two previously used development papers; no independent human review',
              'paper_count':2, 'image_query_count':len(cases), 'candidate_scope':'all indexed papers; no gold filter',
              'text_metadata':text.metadata, 'visual_metadata':visual.metadata, 'device':args.device, 'cases':[]}
    for case, vec, visual_id, text_id in zip(cases, image_vectors, expected_visual, expected_text):
        query = vec.reshape(1,-1)
        visual_hits = visual._search(query, 5, None, None, time.perf_counter())['hits']
        text_hits = text.search_vector(query, 5)['hits']
        expanded, _ = _expand_located_context(visual_hits, bm25.by_id)
        expanded_text = [h for h in expanded if h.get('evidence_type') in (None, 'text')]
        def compact(hits):
            return [{k:h[k] for k in ('paper_id','chunk_id','page','bbox','rank','figure_number','docling_label','matched_span','matched_text') if k in h} for h in hits]
        output['cases'].append({'id':case['id'], 'image_to_image':{'target':visual_id,'top5':compact(visual_hits),
                                  'hit_at_5':any(h['chunk_id']==visual_id for h in visual_hits)},
            'image_to_text':{'target':text_id,'top5':compact(text_hits),
                             'hit_at_5':any(h['chunk_id']==text_id for h in text_hits)},
            'visual_association_expansion':{'method':'unique same-PDF same-page exact located caption; association remains unverified',
                    'added_text':compact(expanded_text), 'target_found':any(h['chunk_id']==text_id for h in expanded_text)}})
    queries = [('three stages of ULMFiT pretraining language model fine tuning classifier', 'v-048bf54e3c62918316858d40'),
               ('BART 文本填空 噪声变换', 'v-a4e9cf505dc170fd9903bd70')]
    # Exercise an actual later representation without changing the BGE chunk.
    spans = {}
    for rep in text.representations:
        spans.setdefault((rep['paper_id'],rep['chunk_id']), []).append(rep)
    tail = next(group[-1] for group in spans.values() if len(group) > 1 and len(group[-1]['text']) > 100)
    vectors = enc.encode_texts([q for q, _ in queries] + [tail['text']])
    output['text_to_image'] = []
    for (query, target), vector in zip(queries, vectors[:len(queries)]):
        hits = visual._search(vector.reshape(1,-1), 5, None, None, time.perf_counter())['hits']
        output['text_to_image'].append({'query':query, 'target':target,
            'top5':[{'paper_id':h['paper_id'],'chunk_id':h['chunk_id'],'rank':h['rank']} for h in hits],
            'hit_at_5':any(h['chunk_id']==target for h in hits)})
    tail_hits = text.search_vector(vectors[-1].reshape(1,-1), 5)['hits']
    output['long_text_self_query'] = {'description':'tail text used as query; diagnostic, not held-out relevance',
        'target':{'paper_id':tail['paper_id'],'chunk_id':tail['chunk_id'],'span':[tail['start'],tail['end']]},
        'top5':[{'paper_id':h['paper_id'],'chunk_id':h['chunk_id'],'matched_span':h['matched_span']} for h in tail_hits],
        'hit_at_5':any(h['paper_id']==tail['paper_id'] and h['chunk_id']==tail['chunk_id'] and h['matched_span']==[tail['start'],tail['end']] for h in tail_hits)}
    query = 'Text Infilling Poisson single MASK token'
    hits = bm25.search(query, mode='bm25', top_k=5)['hits']
    target = 'v-6eb2eb027e9afb995f1da312'
    output['text_to_text'] = {'query':query,'target':target,
        'top5':[{'paper_id':h['paper_id'],'chunk_id':h['chunk_id']} for h in hits],
        'hit_at_5':any(h['chunk_id']==target for h in hits)}
    output['elapsed_seconds'] = time.monotonic()-started
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out), 'image_to_text':[c['image_to_text']['hit_at_5'] for c in output['cases']],
                      'image_to_image':[c['image_to_image']['hit_at_5'] for c in output['cases']],
                      'text_to_image':[c['hit_at_5'] for c in output['text_to_image']],
                      'text_to_text':output['text_to_text']['hit_at_5'], 'elapsed_seconds':output['elapsed_seconds']}))


if __name__ == '__main__':
    main()
