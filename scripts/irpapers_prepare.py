"""Stream-audit and prepare IRPAPERS; never print page-image base64."""
import argparse
import base64
import csv
import hashlib
import io
import json
import re
import time
from collections import Counter
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--hf-revision',default='7d8ca2f6dd9efded3e27013d15782d584f93e9da')
    args=p.parse_args(); source=Path(args.source); out=Path(args.output)
    if out.exists(): raise ValueError('Choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=False); (out/'images').mkdir()
    csv.field_size_limit(2**31-1)
    names=['irpapers-docs.csv','irpapers-queries.csv','irpapers-visual-queries.csv']
    files={n:{'bytes':(source/n).stat().st_size,'sha256':sha(source/n)} for n in names}
    print('source hashes complete',flush=True)
    from PIL import Image,ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    pages={}; pairs={}; papers=set(); counts=Counter(); image_errors=Counter(); begin=time.monotonic()
    with (source/names[0]).open(encoding='utf-8-sig',newline='') as f, (out/'pages.jsonl').open('x',encoding='utf8',newline='\n') as dest:
        for row in csv.DictReader(f):
            counts['rows']+=1
            pid=(row.get('pdf_id_x') or '').strip(); number=(row.get('page_number') or '').strip(); page=(row.get('dataset_id') or '').strip()
            if not re.fullmatch(r'\d+',pid) or not re.fullmatch(r'\d+',number) or not re.fullmatch(r'\d+_\d+',page):
                counts['invalid_identity']+=1; continue
            number=int(number); key=(pid,number)
            if page in pages: counts['duplicate_dataset_id']+=1; continue
            if key in pairs: counts['duplicate_paper_page']+=1; continue
            if page!=f'{pid}_{number}': counts['dataset_id_mismatch']+=1
            text=row.get('transcription') or ''
            if not text.strip(): counts['empty_transcription']+=1
            image_data=None; image_meta={}; raw=row.get('base64_str') or ''
            try:
                image_data=base64.b64decode(raw,validate=True)
                with Image.open(io.BytesIO(image_data)) as image:
                    if image.format not in {'PNG','JPEG'} or image.width*image.height>100_000_000:
                        raise ValueError('unsupported_image')
                    image.load()
                    width,height,fmt=image.width,image.height,image.format
                ext='png' if fmt=='PNG' else 'jpg'
                relative=f'images/{page}.{ext}'
                with (out/relative).open('xb') as img: img.write(image_data)
                image_meta={'image_path':relative,'image_sha256':hashlib.sha256(image_data).hexdigest(),
                            'image_format':fmt,'width':width,'height':height,'image_bytes':len(image_data)}
                counts['image_valid']+=1
                if row.get('base64_bytes') and int(float(row['base64_bytes']))!=len(raw):
                    counts['encoded_length_mismatch']+=1
            except Exception as exc:
                counts['image_invalid']+=1; image_errors[type(exc).__name__]+=1
                image_meta={'image_status':'invalid','error_type':type(exc).__name__}
            del raw,image_data
            item={'page_id':page,'paper_id':'irp-'+pid,'pdf_id':pid,'page_number':number,
                  'title':row.get('pdf_title') or '', 'text':text,
                  'text_source':'IRPAPERS model transcription; not Docling or verified PDF text',**image_meta}
            dest.write(json.dumps(item,ensure_ascii=False)+'\n')
            pages[page]={'pdf_id':pid,'page_number':number,'has_text':bool(text.strip()),'has_image':bool(image_meta.get('image_path'))}
            pairs[key]=page; papers.add(pid)
            if counts['rows']%200==0: print('pages audited',counts['rows'],'seconds',round(time.monotonic()-begin,1),flush=True)
    query_sets={}; all_rows={}
    for label,name in [('standard',names[1]),('visual',names[2])]:
        with (source/name).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        all_rows[label]=rows; c=Counter(); normalized=Counter(); exact=Counter()
        for row in rows:
            page=(row.get('dataset_id') or '').strip(); pid=(row.get('pdf_id') or '').strip(); num=(row.get('page_number') or '').strip()
            q=(row.get('question') or '').strip(); a=(row.get('answer') or '').strip()
            if not q: c['empty_question']+=1
            if not a: c['empty_answer']+=1
            x=pages.get(page)
            if x is None: c['missing_page']+=1
            elif str(x['pdf_id'])!=pid or str(x['page_number'])!=num: c['page_identity_conflict']+=1
            else:
                c['page_present']+=1
                if x['has_text']: c['text_present']+=1
                if x['has_image']: c['image_present']+=1
            normalized[' '.join(q.casefold().split())]+=1; exact[(page,' '.join(q.casefold().split()))]+=1
        c['duplicate_question_excess']=sum(v-1 for v in normalized.values() if v>1)
        c['duplicate_page_question_excess']=sum(v-1 for v in exact.values() if v>1)
        query_sets[label]={'rows':len(rows),'paper_count':len({r.get('pdf_id') for r in rows}),**c}
    norm=lambda r:' '.join((r.get('question') or '').casefold().split())
    s={(r.get('dataset_id'),norm(r)) for r in all_rows['standard']}; v={(r.get('dataset_id'),norm(r)) for r in all_rows['visual']}
    overlap={'same_page_and_question_unique':len(s&v),
             'same_question_unique':len({norm(r) for r in all_rows['standard']} & {norm(r) for r in all_rows['visual']}),
             'same_page_unique':len({r.get('dataset_id') for r in all_rows['standard']} & {r.get('dataset_id') for r in all_rows['visual']}),
             'same_paper_unique':len({r.get('pdf_id') for r in all_rows['standard']} & {r.get('pdf_id') for r in all_rows['visual']})}
    audit={'source':'https://huggingface.co/datasets/weaviate/IRPAPERS','hf_revision_observed':args.hf_revision,
           'revision_verification':'compare local SHA-256 to HF LFS SHA for docs; small CSVs locally hashed',
           'license':'MIT dataset card; underlying paper rights may differ', 'files':files,
           'pages':dict(counts),'papers':len(papers),'image_errors':dict(image_errors),'queries':query_sets,'overlap':overlap,
           'elapsed_seconds':time.monotonic()-begin,'pages_sha256':sha(out/'pages.jsonl')}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
