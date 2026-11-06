"""Score an already-frozen small image-query fixture against full local indices."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',required=True); p.add_argument('--fixture-sha256',required=True)
    p.add_argument('--output',required=True); p.add_argument('--index',required=True)
    p.add_argument('--clip-text-index',required=True); p.add_argument('--visual-index',required=True)
    p.add_argument('--visual-root',required=True); p.add_argument('--visual-python',required=True)
    p.add_argument('--image-model',required=True); p.add_argument('--text-model',required=True)
    p.add_argument('--device',default='cpu'); args=p.parse_args()
    raw=Path(args.fixture).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.fixture_sha256: raise ValueError('Frozen fixture hash mismatch')
    fixture=json.loads(raw); out=Path(args.output)
    if out.exists(): raise ValueError('Choose a new output file')
    for key,hash_key in [('query','query_sha256'),('negative_query','negative_sha256')]:
        if hashlib.sha256(Path(fixture[key]).read_bytes()).hexdigest()!=fixture[hash_key]:
            raise ValueError('Frozen query checksum mismatch')
    from paper_agent.visual_retrieval import SubprocessEncoder,VisualRetriever,fuse_visual_hits
    from paper_agent.clip_text_index import ClipTextRetriever
    from paper_agent.pipeline import _near_duplicate_anchors,_expand_located_context
    from paper_agent.retrieval import Retriever
    enc=SubprocessEncoder(args.visual_python,args.image_model,args.text_model,args.device)
    visual=VisualRetriever(args.visual_index,args.visual_root,enc)
    text=Retriever(args.index)
    clip_text=ClipTextRetriever(args.clip_text_index,enc,text.metadata['corpus_hash'])
    target=[x for x in visual.evidence if x.get('paper_id')==fixture['paper_id'] and x.get('page')==fixture['page']
            and x.get('figure_number')==fixture['figure_number'] and x.get('pdf_sha256')==fixture['pdf_sha256']]
    if len(target)!=1: raise ValueError('Gold figure missing or ambiguous in evaluated index')
    target=target[0]
    captions=[x for x in text.by_id.values() if x.get('paper_id')==fixture['paper_id'] and x.get('page')==fixture['page']
              and x.get('pdf_sha256')==fixture['pdf_sha256'] and x.get('docling_label')=='caption'
              and x.get('text','').strip()==target.get('caption','').strip()]
    if len(captions)!=1: raise ValueError('Gold caption missing or ambiguous in evaluated index')
    caption=captions[0]
    vectors=enc.encode_images([fixture['query'],fixture['negative_query']])
    output={'fixture_sha256':args.fixture_sha256,'scope':fixture['scope'],'target_figure':target['chunk_id'],
            'target_caption':caption['chunk_id'],'index_coverage':{'text':len(text.chunks),'visual':len(visual.evidence)},
            'candidate_scope':'all indexed entities; no target or paper filters','images':[]}
    def ids(hits):
        return [{k:x[k] for k in ('paper_id','chunk_id','rank','visual_score','clip_text_score','matched_span') if k in x} for x in hits]
    for label,vector in zip(('positive','off_corpus_negative'),vectors):
        query=vector.reshape(1,-1)
        figures=visual._search(query,20,None,None,time.perf_counter())['hits']
        all_text=clip_text.search_vector(query,5)['hits']
        cap_text=clip_text.search_vector(query,5,kind='caption')['hits']
        expanded,_=_expand_located_context(figures[:5],text.by_id)
        anchor=_near_duplicate_anchors([figures],1,2)
        output['images'].append({'id':label,'image_to_image_top5':ids(figures[:5]),'direct_text_top5':ids(all_text),
            'direct_caption_top5':ids(cap_text),'associated_text_ids':[x['chunk_id'] for x in expanded if x.get('evidence_type')=='text'],
            'anchor':ids(anchor or []), 'anchor_status':'candidate' if anchor else 'uncertain',
            'figure_hit_at_5':any(x['chunk_id']==target['chunk_id'] for x in figures[:5]) if label=='positive' else None,
            'direct_text_hit_at_5':any(x['chunk_id']==caption['chunk_id'] for x in all_text) if label=='positive' else None,
            'direct_caption_hit_at_5':any(x['chunk_id']==caption['chunk_id'] for x in cap_text) if label=='positive' else None,
            'expanded_caption_found':any(x['chunk_id']==caption['chunk_id'] for x in expanded) if label=='positive' else None})
    tv=enc.encode_texts(fixture['text_queries']); output['text_to_image']=[]
    for query,vector in zip(fixture['text_queries'],tv):
        clip=visual._search(vector.reshape(1,-1),20,None,None,time.perf_counter())['hits']
        captions=visual.search_caption(query,top_k=20)['hits']
        fused=fuse_visual_hits(clip,captions,5)
        output['text_to_image'].append({'query':query,'clip_top5':ids(clip[:5]),'fused_top5':ids(fused),
            'clip_hit_at_5':any(x['chunk_id']==target['chunk_id'] for x in clip[:5]),
            'fused_hit_at_5':any(x['chunk_id']==target['chunk_id'] for x in fused)})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'output':str(out),'images':[{k:x[k] for k in ('id','figure_hit_at_5','direct_text_hit_at_5','direct_caption_hit_at_5','expanded_caption_found','anchor_status')} for x in output['images']],
                      'text_to_image':[{k:x[k] for k in ('clip_hit_at_5','fused_hit_at_5')} for x in output['text_to_image']]}))


if __name__=='__main__': main()
