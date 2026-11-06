from paper_agent.citation_guards import check_claim_numbers


def test_local_conditions_require_standalone_and_test_split_evidence():
    from paper_agent.citation_guards import check_claim_conditions
    evidence = [{'paper_id':'p','chunk_id':'table','evidence_type':'table','text':'24.0'},
                {'paper_id':'p','chunk_id':'caption','evidence_type':'text','text':'Test set perplexity.'}]
    def claim(text, *ids):
        return {'key':'k','text':text,'citations':[{'paper_id':'p','chunk_id':cid} for cid in ids]}
    assert check_claim_conditions([claim('With SBERT, this is reduced to 5 seconds.','caption')],evidence)[0]['status']=='context_dependent'
    assert check_claim_conditions([claim('The test perplexity is 24.0.','table')],evidence)[0]['status']=='test_condition_unknown'
    assert check_claim_conditions([claim('The test perplexity is 24.0.','table','caption')],evidence)[0]['status']=='ok'


def claim(text, *ids):
    return {'key':'k', 'text':text, 'citations':[{'paper_id':'p','chunk_id':i} for i in ids]}


def row(i, text, kind='text'):
    return {'paper_id':'p','chunk_id':i,'text':text,'evidence_type':kind}


def test_numeric_coverage_is_per_claim_not_previous_claim():
    ev = [row('first','10,000 sentences'), row('second','65 hours to 5 seconds')]
    claims = [claim('10,000 sentences', 'first'), claim('10,000 sentences take 5 seconds','second')]
    assert [x['status'] for x in check_claim_numbers(claims, ev)] == ['covered','missing']
    assert check_claim_numbers([claim('10,000 sentences take 5 seconds','first','second')],ev)[0]['status'] == 'covered'


def test_numeric_commas_units_percentage_and_scales():
    ev = [row('a','10,000 sentences take 5 seconds and score 81.4% and 0.001.')]
    assert check_claim_numbers([claim('10000 sentences in 5秒, 81.4 percent, 1e-3','a')],ev)[0]['status'] == 'covered'
    assert check_claim_numbers([claim('10000 sentences in 5 hours','a')],ev)[0]['status'] == 'missing'
    assert check_claim_numbers([claim('81.4 years','a')],ev)[0]['status'] == 'missing'
    assert check_claim_numbers([claim('百分之81.4','a')],ev)[0]['status'] == 'covered'


def test_missing_wrong_ids_and_visual_are_not_fabricated_support():
    ev = [row('a','It took 5 seconds.'),row('figure','caption is not image data','figure')]
    assert check_claim_numbers([claim('10,000 in 5 seconds','a')],ev)[0]['status'] == 'missing'
    assert check_claim_numbers([claim('5 seconds','unknown')],ev)[0]['status'] == 'invalid_reference'
    assert check_claim_numbers([claim('10,000 in 5 seconds','figure')],ev)[0]['status'] == 'unknown_visual'
    assert check_claim_numbers([claim('10,000 in 5 seconds','a','figure')],ev)[0]['status'] == 'unknown_visual'


def test_only_complete_cited_paper_identifier_is_not_a_numeric_claim():
    pid='2020.acl-main.703'
    ev=[{'paper_id':pid,'chunk_id':'title','text':'BART title','evidence_type':'text'}]
    def c(text): return {'key':'k','text':text,'citations':[{'paper_id':pid,'chunk_id':'title'}]}
    assert check_claim_numbers([c('论文 2020.acl-main.703 的标题是 BART title')],ev)[0]['status'] == 'no_explicit_number'
    assert check_claim_numbers([c('论文 2020.acl-main.703 在 2020 年发布')],ev)[0]['status'] == 'missing'
    assert check_claim_numbers([c('论文 2020.acl-main.703x 的标题是 BART title')],ev)[0]['status'] == 'missing'


def test_collection_condition_cannot_be_replaced_by_embedding_count():
    from paper_agent.citation_guards import check_question_collection_conditions
    question='How long among 10,000 sentences?'
    claims=[claim('It takes 5 seconds for 10,000 sentence embeddings','a'),
            claim('Among 10,000 sentences it takes 5 seconds','a')]
    assert [r['status'] for r in check_question_collection_conditions(question,claims)] == ['collection_condition_missing','ok']


def test_collection_guard_does_not_treat_figure_and_paper_ids_as_collection_sizes():
    from paper_agent.citation_guards import check_question_collection_conditions
    assert check_question_collection_conditions('In Table 1 of paper 2020.acl-main.703, what value?',
                                                [claim('The value is 6.61','a')]) == []
