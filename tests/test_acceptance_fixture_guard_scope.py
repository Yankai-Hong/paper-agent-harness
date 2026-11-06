import json
from pathlib import Path

from paper_agent.citation_guards import check_question_collection_conditions
from paper_agent.decomposition import unknown_literal_metadata_id


def test_post_v3_guards_do_not_trigger_on_recorded_nine_question_texts():
    fixture=json.loads((Path(__file__).resolve().parents[1]/'data/eval/multimodal-holdout-v3/cases.json').read_text(encoding='utf8'))
    catalog=[{'paper_id':p['paper_id'],'title':p['title']} for p in fixture['papers']]
    for case in fixture['cases']:
        assert not unknown_literal_metadata_id(case['question'],catalog), case['id']
        assert not check_question_collection_conditions(case['question'],[{'key':'probe','text':'value 5 seconds'}]), case['id']
