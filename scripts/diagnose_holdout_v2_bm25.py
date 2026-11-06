"""Post-run BM25-only diagnostics; never a substitute for first-run traces.

The first run's trace projection excluded route names containing digits, so
its BM25 route ranks cannot be recovered. This separately runs the exact frozen
questions through real lexical indices with no paper filter and, independently,
with oracle paper filters. It does not call the generative model.
"""
import json
from pathlib import Path
from paper_agent.pipeline import _figure_refs, _requested_figure
from paper_agent.retrieval import Retriever
from paper_agent.visual_retrieval import SubprocessEncoder, VisualRetriever

gold = {
 'v201':{'v-aa353350eacefb1f8c95708d','v-2fbf5a8801a0edf308bc1a98'},
 'v202':{'v-a95e7d5f2b33cf30e5636ebd','v-bcb662d63ac62b1b64676acc'},
 'v203':{'v-bd5fb1a3c155e453b94417a8'},
 'v204':{'v-7c7dfc257c4fed7da8703472'},
 'v205':{'v-2a91c8d9e30bafb52e70bf11','v-de9db05380e0050f74462976','v-9430f2cd13d4c5067c84ad78'},
 'v206':{'v-9d13e4571a7ff69ff1dd7cd0','v-6c0fc2e992a434c5407ea63a','v-69faa7d5902f8ccc38b011c3'},
 'v207':{'v-aaa51bde12f46c1511122763'},
 'v208':{'v-aa353350eacefb1f8c95708d','v-2a91c8d9e30bafb52e70bf11'},
 'v209':{'v-aa353350eacefb1f8c95708d'},
}
fixture = json.loads(Path('data/eval/multimodal-holdout-v2/cases.json').read_text(encoding='utf8'))
text = Retriever('indexes/multimodal-holdout-v2-text')
clip = SubprocessEncoder('.venv-mm-20261030/Scripts/python.exe',
    'models/multimodal-v1/clip-image-v1','models/multimodal-v1/clip-text-v1','cuda:0')
visual = VisualRetriever('indexes/multimodal-holdout-v2-clip',
    'data/raw/multimodal-holdout-v2-docling/crops',clip)
result = {'note':'post-run original-question BM25 diagnostics, not first-run route results', 'cases':{}}
for case in fixture['cases']:
    cid=case['id']
    if cid not in gold: continue
    papers=sorted({e['paper_id'] for e in case['evidence']})
    number=_requested_figure(case['question']) if len(_figure_refs(case['question'])) == 1 else None
    routes={}
    for label, allowed in [('unscoped',None),('paper_oracle',papers)]:
        t=text.search(case['question'], mode='bm25', top_k=5, candidate_k=30, paper_ids=allowed)
        v=visual.search_caption(case['question'],top_k=5,paper_ids=allowed,figure_number=None)
        for kind, found in [('text_bm25',t),('caption_bm25',v)]:
            refs=[{'paper_id':x['paper_id'],'chunk_id':x['chunk_id']} for x in found['hits']]
            routes[kind+'_'+label]={'top5':refs,'hit_at_5':any(r['chunk_id'] in gold[cid] for r in refs)}
    if number:
        for label, allowed in [('figure_oracle',None),('joint_oracle',papers)]:
            found=visual.search_caption(case['question'],top_k=5,paper_ids=allowed,figure_number=number)
            refs=[{'paper_id':x['paper_id'],'chunk_id':x['chunk_id']} for x in found['hits']]
            routes['caption_bm25_'+label]={'top5':refs,'hit_at_5':any(r['chunk_id'] in gold[cid] for r in refs)}
    result['cases'][cid]=routes
names=sorted({name for routes in result['cases'].values() for name in routes})
result['summary']={name:{'hits':sum(bool(r[name]['hit_at_5']) for r in result['cases'].values() if name in r),
                         'denominator':sum(name in r for r in result['cases'].values())} for name in names}
output=Path('runs/multimodal-holdout-v2-eval-20261031/bm25-postrun-diagnostic.json')
with output.open('x',encoding='utf8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps(result['summary'],ensure_ascii=False))
