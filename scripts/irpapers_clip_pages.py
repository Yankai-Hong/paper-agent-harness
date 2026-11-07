"""Encode dataset pages in bounded batches in the isolated CLIP interpreter."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True); args=p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    req=json.loads(Path(args.request).read_text(encoding='utf8'))
    root=Path(req['prepared']); out=Path(req['output'])
    if out.exists(): raise ValueError('Refusing to overwrite page index')
    out.mkdir(parents=True,exist_ok=False)
    pages=[json.loads(x) for x in (root/'pages.jsonl').open(encoding='utf8') if x.strip()]
    eligible=[x for x in pages if x.get('image_path')]
    if not eligible: raise ValueError('No decodable page images')
    model=SentenceTransformer(req['image_model'],device=req['device'],local_files_only=True)
    if req['device'].startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    vectors_file=None; dimension=None; mapping=[]; start=time.monotonic()
    for start_index in range(0,len(eligible),req['batch_size']):
        group=eligible[start_index:start_index+req['batch_size']]; images=[]
        for page in group:
            path=(root/page['image_path']).resolve(strict=True)
            if not path.is_relative_to(root.resolve()) or sha(path)!=page['image_sha256']:
                raise ValueError('Page image path or hash mismatch')
            with Image.open(path) as img:
                img.load(); images.append(img.convert('RGB'))
        vectors=np.asarray(model.encode(images,batch_size=req['batch_size'],convert_to_numpy=True,
                           show_progress_bar=False,normalize_embeddings=True),dtype=np.float32)
        for image in images: image.close()
        if vectors.ndim!=2 or len(vectors)!=len(group) or not np.isfinite(vectors).all(): raise ValueError('Invalid page vectors')
        norms=np.linalg.norm(vectors,axis=1)
        if np.any(norms<=0): raise ValueError('Zero page vector')
        vectors=np.ascontiguousarray(vectors/norms[:,None])
        if vectors_file is None:
            dimension=vectors.shape[1]
            vectors_file=np.lib.format.open_memmap(out/'vectors.npy',mode='w+',dtype=np.float32,
                                                   shape=(len(eligible),dimension))
        if dimension!=vectors.shape[1]: raise ValueError('Page vector dimension changed')
        vectors_file[start_index:start_index+len(group)]=vectors
        vectors_file.flush()
        for x in group:
            mapping.append({k:x[k] for k in ('page_id','paper_id','pdf_id','page_number','image_path','image_sha256','image_format','width','height')})
        if len(mapping)%200 < req['batch_size']: print('pages encoded',len(mapping),'of',len(eligible),flush=True)
    del vectors_file
    with (out/'mapping.jsonl').open('x',encoding='utf8') as f:
        for x in mapping: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    meta={'schema':1,'kind':'IRPAPERS_page_image_CLIP_not_figure_crop','count':len(mapping),
          'source_pages':len(pages),'missing_images':len(pages)-len(eligible),'dimension':dimension,
          'metric':'cosine','normalization':'l2','preprocessing':'PIL RGB; pinned SentenceTransformer CLIP image processor',
          'model_provenance':req['model_provenance'],'plan_sha256':req['plan_sha256'],'pages_sha256':sha(root/'pages.jsonl'),
          'vectors_sha256':sha(out/'vectors.npy'),'mapping_sha256':sha(out/'mapping.jsonl'),
          'device':req['device'],'batch_size':req['batch_size'],'elapsed_seconds':time.monotonic()-start,
          'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if req['device'].startswith('cuda') else None}
    try:
        import psutil
        meta['process_rss_bytes_final']=psutil.Process().memory_info().rss
    except ImportError: meta['process_rss_bytes_final']=None
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__': main()
