"""Verify delivered local docs, frozen retrieval outputs and test report without API."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from paper_agent.storage import read_jsonl, write_json
from paper_agent.evaluation import retrieval_metrics


root = Path(__file__).resolve().parents[1]
documents = [root/'README.md', *sorted((root/'docs').glob('*.md'))]
broken = []
for document in documents:
    for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
        if '://' not in target and not target.startswith('#'):
            path = target.split('#', 1)[0]
            if not (document.parent/path).exists():
                broken.append(f'{document.relative_to(root)} -> {target}')
if broken:
    raise ValueError(f'Broken local documentation links: {broken}')
queries = read_jsonl(root/'data/qasper/retrieval.jsonl')
verified = {}
for mode in ['bm25', 'dense', 'hybrid', 'rerank']:
    directory = root/'reports/acceptance/retrieval-v2'/mode
    predictions = read_jsonl(directory/'predictions.jsonl')
    recorded = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    recomputed = retrieval_metrics(queries, predictions)
    for key in ['Recall@5', 'Hit@5', 'errors', 'missing']:
        if recomputed[key] != recorded[key]:
            raise ValueError(f'Metric mismatch for {mode}/{key}')
    verified[mode] = {key: recomputed[key] for key in ['Recall@5', 'Hit@5', 'errors', 'missing']}
test_xml = ET.parse(root/'reports/acceptance-tests.xml')
suites = list(test_xml.getroot().iter('testsuite'))
test_counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ['tests', 'errors', 'failures', 'skipped']}
if test_counts['errors'] or test_counts['failures']:
    raise ValueError('Test report includes failures')
manifest = json.loads((root/'data/dialogues-draft/manifest.json').read_text(encoding='utf-8'))
assert len(read_jsonl(root/'data/dialogues-draft/dialogues.jsonl')) == 120
assert manifest['valid_for_effectiveness_claims'] is False
write_json(root/'reports/acceptance/delivery-verification.json', {'tests': test_counts,
    'checked_document_files': len(documents), 'broken_local_links': broken,
    'retrieval_metrics_recomputed_from_saved_predictions': verified,
    'dialogue_draft_status': manifest['status'],
    'verification_api_calls': 0,
    'not_verified': ['independently reviewed multi-turn gold', 'human-reviewed semantic support',
                     'generalized long-term memory quality; only limited development experiments run']})
print('Documentation links, saved retrieval metrics, test report and draft status verified.')
