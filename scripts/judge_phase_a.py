"""Run existing answer-quality judge after the full matrix; no label promotion."""
import subprocess
import sys
import time
import json
from pathlib import Path
from paper_agent.storage import write_json


def main():
    root=Path('reports/history/development/phase-a/full')
    runs=[(root/'memory','data/eval/v0.6/memory.jsonl')]
    runs += [(root/'ablations'/name,'data/eval/v0.3/reflect-subset.jsonl') for name in
             ['reflect-enabled','reflect-disabled','reflect-retrieve-again','reflect-no-second']]
    runs += [(root/'ablations'/name,'data/eval/v0.3/memory-subset.jsonl') for name in
             ['memory-full','memory-bounded','memory-semantic','memory-recall']]
    runs += [(root/'e2e','data/eval/v0.6/dialogs.jsonl')]
    records=[]
    for run,data in runs:
        deadline=time.monotonic()+7200
        while not (run/'per_case.jsonl').exists() and not (run/'predictions.jsonl').exists():
            jobs=json.loads((root/'jobs.json').read_text(encoding='utf-8'))
            job_name='ablations' if 'ablations' in run.parts else run.name
            if any(j['name']==job_name for j in jobs) or time.monotonic()>deadline:
                break
            time.sleep(5)
        started=time.time()
        args=[sys.executable,'scripts/judge_acceptance.py',str(run),data,'--workers','2']
        with (root/(run.name+'-quality.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        records.append({'command':args,'returncode':result.returncode,'elapsed_seconds':time.time()-started})
        write_json('reports/history/development/phase-a/quality-jobs.json',records)
        print(run.name,result.returncode,flush=True)


if __name__=='__main__': main()
