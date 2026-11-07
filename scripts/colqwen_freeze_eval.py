"""Freeze a real development-evaluated ColQwen candidate configuration."""
import argparse
import json
from pathlib import Path

from colqwen_smoke import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-summary', required=True)
    p.add_argument('--encoding-contract', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    summary = json.loads(Path(args.development_summary).read_text(encoding='utf-8'))
    contract = json.loads(Path(args.encoding_contract).read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or summary.get('partition') != 'dev' or contract.get('partition') != 'dev':
        raise ValueError('A completed real development evaluation is required')
    summary_name = str(Path(args.development_summary))
    if (not summary_name.endswith('.summary.json')
            or sha(summary_name[:-len('.summary.json')]) != summary.get('results_sha256')
            or sha(args.encoding_contract) != summary.get('encoding_contract_sha256')
            or summary.get('evaluation_script_sha256') != sha('scripts/colqwen_candidate_eval.py')
            or summary.get('store_code_sha256') != sha('src/paper_agent/resumable_colqwen.py')
            or summary.get('maxsim_code_sha256') != sha('src/paper_agent/colqwen_store.py')
            or summary.get('maxsim_batch_code_sha256') != sha('src/paper_agent/maxsim_batch.py')
            or contract.get('script_sha256') != sha('scripts/colqwen_encode_candidates.py')):
        raise ValueError('Development results or encoding contract mismatch')
    if 'standard' not in summary.get('metrics', {}) or 'visual' not in summary['metrics']:
        raise ValueError('Both development groups must be evaluated')
    plan = {'status': 'frozen_before_test_evaluation', 'development_summary': str(Path(args.development_summary)),
            'development_summary_sha256': sha(args.development_summary),
            'candidate_plan_sha256': contract['candidate_plan_sha256'], 'candidate_results_sha256': contract['candidate_results_sha256'],
            'model_revision': contract['model_revision'], 'base_revision': contract['base_revision'],
            'weight_sha256': contract['weight_sha256'], 'max_visual_tokens': contract['max_visual_tokens'],
            'dimension': contract['dimension'], 'device': contract['device'], 'runtime_versions': contract['runtime_versions'],
            'encoder_script_sha256': contract['script_sha256'], 'evaluation_script_sha256': summary['evaluation_script_sha256'],
            'store_code_sha256': summary['store_code_sha256'], 'maxsim_code_sha256': summary['maxsim_code_sha256'],
            'maxsim_batch_code_sha256': summary['maxsim_batch_code_sha256'], 'score_device': summary['score_device'],
            'cache_pages': summary['cache_pages'], 'scoring_torch_version': summary['scoring_torch_version'],
            'method': 'MaxSim within frozen candidates, remaining text pages in existing order',
            'strict_blind_test': False, 'reason': 'historical baseline test aggregates were previously inspected'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, sort_keys=True, indent=2)
    print(json.dumps({'output': str(output), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
