"""Frozen full regression. Existing outputs are never reused as current results."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from paper_agent.storage import write_json

ROOT = Path('reports/phase-a/full')


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    sources = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['src', 'scripts', 'tests', 'configs', 'data/eval/v0.6']
               for p in Path(directory).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.jsonl'}}
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()
    write_json(ROOT/'freeze.json', {'utc':datetime.now(timezone.utc).isoformat(), 'commit':git('rev-parse','HEAD'),
        'workspace':git('status','--short'), 'hashes':sources, 'note':'Dirty source identified by individual SHA-256; .env credentials excluded'})
    base = ['-m', 'paper_agent.cli', 'eval']
    common = ['--max-tokens','4096','--thinking','disabled','--workers','4']
    pipe = ['--index','indexes/bge','--mode','bm25', *common]
    jobs = [('tests', ['-m','pytest','-q','--junitxml='+str(ROOT/'tests.xml')]),
        ('intent',base+['intent','data/eval/v0.6/intent.jsonl','--output',str(ROOT/'intent'),*common]),
        ('rewrite',base+['rewrite','data/eval/v0.6/query_rewrite.jsonl','--output',str(ROOT/'rewrite'),*common]),
        ('retrieval',['scripts/run_retrieval_suite.py','--reuse-index','--output',str(ROOT/'retrieval')]),
        ('boundary',base+['boundary','data/eval/v0.6/boundary.jsonl','--output',str(ROOT/'boundary'),*pipe]),
        ('memory',base+['memory','data/eval/v0.6/memory.jsonl','--output',str(ROOT/'memory'),*pipe]),
        ('ablations',base+['ablation','--config','configs/phase-a.json','--output',str(ROOT/'ablations')]),
        ('e2e',base+['e2e','data/eval/v0.6/dialogs.jsonl','--output',str(ROOT/'e2e'),*pipe]),
        ('claims',['scripts/evaluate_claims.py',str(ROOT/'e2e')])]
    results = []
    for name, args in jobs:
        print('START '+name, flush=True)
        started = time.time()
        with (ROOT/(name+'.log')).open('w',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
        results.append({'name':name,'command':[sys.executable,*args],'started_utc':datetime.fromtimestamp(started,timezone.utc).isoformat(),
            'elapsed_seconds':time.time()-started,'returncode':result.returncode})
        write_json(ROOT/'jobs.json',results)
        print('END '+name+' '+str(result.returncode),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT))
    args = parser.parse_args()
    ROOT = Path(args.output)
    main()
