"""Fail closed if a full-corpus continuation's model/runtime contract differs."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from colqwen_smoke import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dev','test','prepared','base','adapter','config-hashes','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args(); dev=Path(a.dev); test=Path(a.test)
    c1=json.loads((dev/'contract.json').read_text(encoding='utf-8'))
    c2=json.loads((test/'contract.json').read_text(encoding='utf-8'))
    fields=('base_revision','model_revision','weight_sha256','max_visual_tokens','dimension',
            'runtime_versions','device','script_sha256','source_pages_sha256','smoke_sha256')
    if any(c1[k]!=c2[k] for k in fields) or sha(Path(a.prepared)/'pages.jsonl')!=c1['source_pages_sha256']:
        raise ValueError('Page source or partition contracts differ')
    actual={name:importlib.metadata.version(name) for name in c1['runtime_versions']}
    if actual!=c1['runtime_versions']: raise ValueError('Current model runtime differs')
    for filename, expected in c1['weight_sha256'].items():
        path=(Path(a.adapter) if filename=='adapter_model.safetensors' else Path(a.base))/filename
        if sha(path)!=expected: raise ValueError('Model weight differs: '+filename)
    config=json.loads(Path(a.config_hashes).read_text(encoding='utf-8'))['sha256']
    for path,expected in config.items():
        if sha(path)!=expected: raise ValueError('Model configuration differs: '+path)
    report={'status':'verified','source_pages_sha256':c1['source_pages_sha256'],
            'model_revision':c1['model_revision'],'base_revision':c1['base_revision'],
            'runtime_versions':actual,'config_files_checked':len(config),'weight_files_checked':len(c1['weight_sha256']),
            'config_hash_manifest_sha256':sha(a.config_hashes),
            'limitation':'Configuration hashes were observed in 2026-11-08, not bound by earlier checkpoint generation.'}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
