"""Hash existing assets and current source, without reading credential contents."""
import subprocess
import ast
from datetime import datetime, timezone
from pathlib import Path
from report_phase_a import sha
from paper_agent.storage import write_json


def main():
    assets = {}
    for directory in ['models','indexes/bge','data/qasper','data/raw','data/eval/v0.6','data/eval/claim-diagnostics-v1']:
        for path in Path(directory).rglob('*'):
            if path.is_file():
                assets[str(path)] = {'bytes':path.stat().st_size,'sha256':sha(path)}
    source = {str(p):sha(p) for directory in ['src','scripts','tests','configs'] for p in Path(directory).rglob('*')
              if p.is_file() and p.suffix in {'.py','.json'}}
    retrieval_current=Path('src/paper_agent/retrieval.py').read_text(encoding='utf-8')
    retrieval_head=subprocess.check_output(['git','show','HEAD:src/paper_agent/retrieval.py'],text=True,encoding='utf-8')
    write_json('reports/phase-a/final-provenance.json', {
        'utc':datetime.now(timezone.utc).isoformat(), 'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'workspace_status':subprocess.check_output(['git','status','--short'],text=True),
        'source_sha256':source,'assets':assets,'credentials':'excluded',
        'result_sha256':{str(p):sha(p) for p in Path('reports/phase-a').rglob('*') if p.is_file()
                        and p.suffix in {'.json','.jsonl','.xml'} and p.name!='final-provenance.json'},
        'concurrent_user_retrieval_edit':{'preserved':True,'AST_matches_HEAD':ast.dump(ast.parse(retrieval_current))==ast.dump(ast.parse(retrieval_head))},
        'model_version_note':'API response exposes model name, not immutable remote weight revision; local BGE weight files hashed.'})
    print('Hashed',len(assets),'asset files and',len(source),'source/config files')


if __name__=='__main__': main()
