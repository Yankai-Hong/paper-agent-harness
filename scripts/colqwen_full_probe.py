"""Compare exact batched MaxSim with the official processor on real pages."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from colqwen_encode_candidates import check_marker
from colqwen_smoke import rows, sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paper_agent.colqwen_store import maxsim
from paper_agent.maxsim_batch import score_maxsim_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('prepared','encoded','dev','baseline','adapter','output'): p.add_argument('--'+x,required=True)
    p.add_argument('--case',default='standard-006'); p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import torch
    from colpali_engine.models import ColQwen2Processor
    source, encoded, dev, root=map(Path,(a.prepared,a.encoded,a.dev,a.baseline))
    config=json.loads((encoded/'contract.json').read_text(encoding='utf-8'))
    if sha(source/'pages.jsonl')!=config['source_pages_sha256']: raise ValueError('Source mismatch')
    case=next(x for x in rows(root/'text-results.jsonl') if x['case_id']==a.case and x['partition']=='dev')
    query_folder=dev/'queries'; expected={'case_id':a.case,'question_sha256':hashlib.sha256(case['question'].encode()).hexdigest()}
    if not check_marker(query_folder,a.case,expected): raise ValueError('Query checkpoint missing')
    marker=json.loads((query_folder/(a.case+'.json')).read_text(encoding='utf-8'))
    with np.load(query_folder/marker['vector_file'],allow_pickle=False) as z:q=np.asarray(z['vectors'],dtype=np.float32)
    docs=[]; page_ids=[]
    for page in rows(source/'pages.jsonl')[:3]:
        key=page['page_id']; folder=encoded/'pages'
        if not check_marker(folder,key,{'page_id':key,'paper_id':page['paper_id'],'image_sha256':page['image_sha256']}):
            raise ValueError('Page checkpoint missing')
        m=json.loads((folder/(key+'.json')).read_text(encoding='utf-8'))
        with np.load(folder/m['vector_file'],allow_pickle=False) as z: docs.append(np.asarray(z['vectors'],dtype=np.float32))
        page_ids.append(key)
    processor=ColQwen2Processor.from_pretrained(a.adapter,local_files_only=True,
                                                 max_num_visual_tokens=config['encoder']['max_visual_tokens'])
    batch=score_maxsim_batch(q,docs,device=a.device)
    single=[maxsim(q,x) for x in docs]
    official=[float(processor.score_multi_vector([torch.from_numpy(q)],[torch.from_numpy(x)],device='cpu')[0,0]) for x in docs]
    diff=max(abs(x-y) for xs,ys in ((batch,single),(batch,official)) for x,y in zip(xs,ys))
    if diff>1e-3: raise ValueError('Official and batched MaxSim disagree')
    report={'case_id':a.case,'page_selection':'first three source pages, independent of gold',
            'page_ids':page_ids,'batch':batch,'single':single,'official':official,'max_abs_difference':diff,
            'scope':'real masked token vectors; small numerical equivalence probe'}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
