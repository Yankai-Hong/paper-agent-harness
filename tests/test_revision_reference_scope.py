"""Scope guard regressions: no historical memory is used as paper evidence."""
import pytest
from paper_agent.pipeline import PaperAgentState
from test_revision_decomposition import make_complex_pipeline


def prior(*turns):
    state=PaperAgentState('s')
    for i,papers in enumerate(turns):
        if isinstance(papers,str): papers=[papers]
        state.messages.extend([
            {'role':'user','turn_id':f'old{i}','content':'讨论 '+' 和 '.join(papers)},
            {'role':'assistant','turn_id':f'old{i}','content':{'claims':[
                {'text':p,'citations':[{'paper_id':p,'chunk_id':'c'}]} for p in papers]}}])
    state.memory.summary['discussed_papers']=list(dict.fromkeys(p for pset in turns for p in ([pset] if isinstance(pset,str) else pset)))
    return state


def nlu(*papers):
    return {'intent':'compare','standalone_query':'accuracy','paper_ids':list(papers),'resolved_entities':{}}


def test_unique_paper_aliases_resolve_explicit_comparison_without_history(tmp_path):
    model={**nlu('paperA','paperB'), 'standalone_query':'Compare AlphaVec and Sentence-BERT'}
    p,searches,_=make_complex_pipeline(tmp_path,nlu=model,decomposition=False)
    a,b,_=p.retriever.by_id.values()
    a.update(title='Deep Contextualized Word Representations',
             text='For this reason, we call them AlphaVec (Advanced vectors) representations.')
    b.update(title='Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks')
    for turn, question in enumerate([
        '比较 Deep Contextualized Word Representations 和 Sentence-BERT 两篇论文的表示粒度',
        '在 AlphaVec 和 Sentence-BERT 两篇论文中，哪一篇的 Figure 2 显示余弦相似度？',
    ]):
        assert p.run(question, prior(), str(turn))['answer']['decision']=='answer'
    assert searches and all(s['paper_ids']==['paperA','paperB'] for _,s in searches)


def test_alias_collision_and_historical_reference_remain_ambiguous(tmp_path):
    from paper_agent.decomposition import catalog_with_literal_aliases, scope_guard
    chunks=[{'paper_id':'paperA','title':'One Paper','text':'We call them AlphaVec (vectors).'},
            {'paper_id':'paperB','title':'Other Paper','text':'We call them AlphaVec (methods).'}]
    catalog=catalog_with_literal_aliases(chunks)
    assert all('alphavec' not in p.get('aliases',[]) for p in catalog)
    assert scope_guard('比较 paperA 和之前那篇',catalog,['paperA','paperB'],prior())=='explicit_targets_unresolved'
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA','paperB'),decomposition=False)
    assert p.run('比较 paperA 和之前那篇',prior(),'t')['answer']['decision']=='clarify'
    assert not searches


def test_title_alias_in_single_paper_possessive_is_local():
    from paper_agent.decomposition import catalog_with_literal_aliases, scope_guard
    catalog=catalog_with_literal_aliases([
        {'paper_id':'p1','title':'BERT: Pre-training of Deep Bidirectional Transformers','text':'abstract'}])
    assert scope_guard('How does BERT evaluate its method?',catalog,['p1'],prior(),
                       'How does BERT evaluate its method?',{}) is None


@pytest.mark.parametrize('question,entities',[
    ('比较 paperA 和 paperB，前者有什么特点，后者有什么特点？',{'前者':'paperA','后者':'paperB'}),
    ('比较 paperB 和 paperA，前者有什么特点，后者有什么特点？',{'前者':'paperB','后者':'paperA'}),
    ('Compare paperA and paperB: the former uses what, the latter uses what?',{'former':'paperA','latter':'paperB'}),
])
def test_same_sentence_former_and_latter_follow_explicit_order(question,entities):
    from paper_agent.decomposition import scope_guard
    catalog=[{'paper_id':'paperA'},{'paper_id':'paperB'},{'paper_id':'paperC'}]
    assert scope_guard(question,catalog,['paperA','paperB'],prior(),None,entities) is None
    assert scope_guard(question,catalog,['paperA'],prior(),None,entities) == 'explicit_targets_unresolved'


@pytest.mark.parametrize('split',[False,True])
def test_explicit_two_paper_order_reaches_pipeline_retrieval(tmp_path,split):
    model={**nlu('paperA','paperB'),'standalone_query':'compare paperA and paperB',
           'resolved_entities':{'前者':'paperA','后者':'paperB'}}
    p, searches, _=make_complex_pipeline(tmp_path,nlu=model,decomposition=split)
    answer=p.run('比较 paperA 和 paperB 的 Figure 2，前者显示什么，后者显示什么？',prior(),'t')['answer']
    assert answer['decision']=='answer' and searches
    assert {(c['citations'][0]['paper_id']) for c in answer['claims']} == {'paperA','paperB'}


@pytest.mark.parametrize('question,ids,entities',[
    ('比较 paperA 和 paperB，前者是什么？',['paperA','paperB'],{'前者':'paperB'}),
    ('比较 paperA 和 paperB，后者是什么？',['paperA','paperB'],{'后者':'paperA'}),
    ('前者是什么？比较 paperA 和 paperB。',['paperA','paperB'],{}),
    ('比较 paperA、paperB、paperC，前者是什么？',['paperA','paperB','paperC'],{}),
    ('比较 paperA 和 paperB。前者是什么？',['paperA','paperB'],{}),
    ('比较 paperA 和 paperB，前者是什么？',['paperA','paperB','paperC'],{}),
])
def test_same_sentence_ordinal_rejects_conflicts_or_missing_order(question,ids,entities):
    from paper_agent.decomposition import scope_guard
    catalog=[{'paper_id':'paperA'},{'paper_id':'paperB'},{'paper_id':'paperC'}]
    assert scope_guard(question,catalog,ids,prior(['paperB','paperC']),None,entities)=='explicit_targets_unresolved'


def test_explicit_historical_former_does_not_use_current_sentence_order():
    from paper_agent.decomposition import scope_guard
    catalog=[{'paper_id':'paperA'},{'paper_id':'paperB'},{'paper_id':'paperC'}]
    state=prior(['paperB','paperC'])
    assert scope_guard('比较 paperA 和先前的前者',catalog,['paperA','paperB'],state,
                       None,{'前者':'paperB'}) is None
    assert scope_guard('比较 paperA 和先前的前者',catalog,['paperA','paperC'],state,
                       None,{'前者':'paperC'})=='explicit_targets_unresolved'


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('restriction',['不包括2024年的数据','排除2024年的实验结果','without data from 2024'])
def test_data_exclusion_remains_question_constraint(tmp_path,split,restriction):
    p,searches,calls=make_complex_pipeline(tmp_path,nlu=nlu('paperA','paperB'),decomposition=split)
    result=p.run('比较 paperA 和 paperB，'+restriction,prior(),'t')
    assert result['answer']['decision']=='answer'
    assert searches and all(restriction in query for query,_ in searches)
    reflect=next(payload for stage,payload in calls if stage=='reflect')
    assert restriction in reflect['question']
    available={(c['paper_id'],c['chunk_id']) for c in next(payload for stage,payload in calls if stage=='generate')['evidence']}
    assert all((r['paper_id'],r['chunk_id']) in available for c in result['answer']['claims'] for r in c['citations'])


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('question',[
    '比较 paperA 和 paperB，排除 paperB',
    '比较 paperA 和 paperB，不包括 paperB 的论文',
    'compare paperA and paperB, excluding paperB',
    '比较 paperA 和 paperB，排除无法确认的对象',
])
def test_paper_or_uncertain_exclusion_clarifies(tmp_path,split,question):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run(question,prior(),'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_data_exclusion_requires_reflection_and_semantic_support(tmp_path,split):
    question='比较 paperA 和 paperB，不包括2024年的数据'
    p,searches,_=make_complex_pipeline(tmp_path/'off',nlu=nlu('paperA','paperB'),decomposition=split,reflect=False)
    result=p.run(question,prior(),'t')['answer']
    assert result['decision']=='clarify' and result['reason_code']=='constraint_verification_unavailable'
    assert not searches
    p,_,calls=make_complex_pipeline(tmp_path/'uncertain',nlu=nlu('paperA','paperB'),decomposition=split,
        verdict=lambda payload:{'decision':'refuse','reason':'conditions not supported'})
    assert p.run(question,prior(),'t')['answer']['decision']=='refuse'
    assert not any(stage=='generate' for stage,_ in calls)


@pytest.mark.parametrize('split',[False,True])
def test_only_data_is_not_only_paper_scope(tmp_path,split):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA','paperB'),decomposition=split)
    question='比较 paperA 和刚才那篇，只使用 2023 年的数据'
    assert p.run(question,prior('paperB'),'t')['answer']['decision']=='answer'
    assert all('只使用 2023 年的数据' in query for query,_ in searches)


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('question',['比较 paperA 跟刚才那篇的准确率','比较 paperA、刚才那篇的准确率'])
def test_historical_reference_must_not_be_dropped_for_any_connector(tmp_path,split,question):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA'),decomposition=split)
    assert p.run(question,prior('paperB'),'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_recent_does_not_skip_latest_unrelated_turn(tmp_path,split):
    state=prior('paperB')
    state.messages.extend([{'role':'user','turn_id':'intervening','content':'请总结一般方法'},
                           {'role':'assistant','turn_id':'intervening','content':{'decision':'refuse','claims':[]}}])
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和刚才那篇',state,'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_previous_turn_is_a_temporal_anchor_for_singular_it(tmp_path,split):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和它（上一轮讨论的论文）的准确率',prior('paperC','paperB'),'t')['answer']['decision']=='answer'
    assert searches


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('rewrite,entities',[
    ('比较 paperA 和 paperC',{}),
    ('比较 paperA 的准确率',{}),
    ('accuracy',{'刚才':'paperC'}),
    ('accuracy',{'它':'paperA'}),
    ('accuracy',{'它':'unknown-fabricated'}),
])
def test_nlu_rewrite_and_entity_must_agree_with_resolved_scope(tmp_path,split,rewrite,entities):
    model={**nlu('paperA','paperB'),'standalone_query':rewrite,'resolved_entities':entities}
    p,searches,_=make_complex_pipeline(tmp_path,nlu=model,decomposition=split)
    assert p.run('比较 paperA 和刚才讨论的那篇论文',prior('paperC','paperB'),'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_explicit_paper_local_demonstrative_does_not_require_history(tmp_path,split):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA'),decomposition=split)
    assert p.run('解释 paperA 这篇论文',prior(),'t')['answer']['decision']=='answer'
    assert searches


@pytest.mark.parametrize('split',[False,True])
def test_scoped_colon_demonstrative_does_not_invent_historical_target(tmp_path,split):
    model={**nlu('paperA'), 'intent':'fact'}
    p,searches,_=make_complex_pipeline(tmp_path,nlu=model,decomposition=split)
    assert p.run('Only paper paperA: 这篇论文使用什么知识库？',prior(),'t')['answer']['decision']=='answer'
    assert searches and all(opts['paper_ids']==['paperA'] for _,opts in searches)
    p,searches,_=make_complex_pipeline(tmp_path/'recent',nlu=model,decomposition=split)
    assert p.run('Only paper paperA: 比较这篇论文和刚才那篇',prior('paperB'),'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_only_local_paper_reference_clarification_can_be_resolved(tmp_path,split):
    model={**nlu('paperA'), 'intent':'fact', 'needs_clarification':True,
           'clarification_reason':'paper_reference'}
    p,searches,_=make_complex_pipeline(tmp_path/'paper',nlu=model,decomposition=split)
    assert p.run('Only paper paperA: 这篇论文使用什么知识库？',prior(),'t')['answer']['decision']=='answer'
    assert searches
    model['clarification_reason']='other'
    p,searches,_=make_complex_pipeline(tmp_path/'other',nlu=model,decomposition=split)
    assert p.run('Only paper paperA: 这篇论文使用什么知识库？',prior(),'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_explicit_artifact_digest_question_reaches_reflect_without_claiming_absence(tmp_path,split):
    model={'intent':'out_of_scope','standalone_query':'model weights checksum',
           'paper_ids':None,'resolved_entities':{},'needs_clarification':False}
    p,searches,calls=make_complex_pipeline(tmp_path/'probe',nlu=model,decomposition=split,
        verdict=lambda payload: {'decision':'refuse','reason':'No adequate artifact evidence'})
    answer=p.run('Only paper paperA: 这篇论文模型权重文件的 SHA-256 是什么？',prior(),'t')['answer']
    assert answer['decision']=='refuse' and searches
    assert any(stage=='reflect' for stage,_ in calls) and not any(stage=='generate' for stage,_ in calls)
    assert all(opts['paper_ids']==['paperA'] for _,opts in searches)
    model['out_of_scope_reason']='not_paper_question'
    p,searches,_=make_complex_pipeline(tmp_path/'other',nlu=model,decomposition=split)
    assert p.run('Only paper paperA: 这篇论文模型权重文件的 SHA-256 是什么？',prior(),'t')['answer']['decision']=='refuse'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_english_internal_it_is_not_automatically_a_second_paper(tmp_path,split):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu('paperA'),decomposition=split)
    assert p.run('How does paperA evaluate its method?',prior(),'t')['answer']['decision']=='answer'
    assert searches and all(opts['paper_ids']==['paperA'] for _,opts in searches)
    p,searches,_=make_complex_pipeline(tmp_path/'compare',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('Compare paperA and it',prior('paperB'),'t')['answer']['decision']=='answer'
    assert searches


@pytest.mark.parametrize('split',[False,True])
def test_singular_reference_requires_unique_grounded_history(tmp_path,split):
    question='比较 paperA 和它的准确率'
    p,searches,_=make_complex_pipeline(tmp_path/'one',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run(question,prior('paperB'),'t')['answer']['decision']=='answer'
    assert searches
    for name,state,ids in [
        ('ambiguous',prior('paperC','paperB'),('paperA','paperB')),
        ('absent',prior(),('paperA','paperB')),
        ('wrong',prior('paperB'),('paperA','paperC')),
        ('dropped',prior('paperB'),('paperA',)),
    ]:
        p,searches,_=make_complex_pipeline(tmp_path/name,nlu=nlu(*ids),decomposition=split)
        assert p.run(question,state,'t')['answer']['decision']=='clarify',name
        assert not searches


@pytest.mark.parametrize('split',[False,True])
def test_compacted_history_cannot_make_generic_pronoun_look_unique(tmp_path,split):
    state=prior('paperB')
    state.archive_ids=['earlier_snapshot']
    state.memory.summary['discussed_papers']=['paperC','paperB']
    p,searches,_=make_complex_pipeline(tmp_path/'generic',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和它的准确率',state,'t')['answer']['decision']=='clarify'
    assert not searches
    state=prior('paperB')
    state.memory.summary['discussed_papers']=['paperC','paperB']
    p,searches,_=make_complex_pipeline(tmp_path/'hint',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和它的准确率',state,'t')['answer']['decision']=='clarify'
    assert not searches
    state=prior('paperB')
    state.archive_ids=['earlier_snapshot']
    state.memory.summary['discussed_papers']=['paperC','paperB']
    p,searches,_=make_complex_pipeline(tmp_path/'recent',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和刚才讨论的那篇',state,'t')['answer']['decision']=='answer'
    assert searches


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('ids,expected',[(('paperA','paperB'),'answer'),(('paperA','paperC'),'clarify'),(('paperA',),'clarify')])
def test_just_now_uses_most_recent_unambiguous_turn(tmp_path,split,ids,expected):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu(*ids),decomposition=split)
    result=p.run('比较 paperA 和刚才讨论的那篇论文',prior('paperC','paperB'),'t')
    assert result['answer']['decision']==expected
    assert bool(searches)==(expected=='answer')
    if expected=='answer':
        assert {c['citations'][0]['paper_id'] for c in result['answer']['claims']}=={'paperA','paperB'}


@pytest.mark.parametrize('split',[False,True])
def test_recent_multiple_targets_and_stale_summary_are_ambiguous(tmp_path,split):
    p,searches,_=make_complex_pipeline(tmp_path/'multiple',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和刚才讨论的那篇论文',prior(['paperB','paperC']),'t')['answer']['decision']=='clarify'
    assert not searches
    state=prior();state.memory.summary['discussed_papers']=['paperC','paperB']
    p,searches,_=make_complex_pipeline(tmp_path/'summary',nlu=nlu('paperA','paperB'),decomposition=split)
    assert p.run('比较 paperA 和刚才讨论的那篇论文',state,'t')['answer']['decision']=='clarify'
    assert not searches


@pytest.mark.parametrize('split',[False,True])
@pytest.mark.parametrize('ids,expected',[(('paperA','paperB'),'answer'),(('paperA','paperC'),'clarify')])
def test_former_respects_explicit_previous_comparison_order(tmp_path,split,ids,expected):
    p,searches,_=make_complex_pipeline(tmp_path,nlu=nlu(*ids),decomposition=split)
    assert p.run('比较 paperA 和前者',prior(['paperB','paperC']),'t')['answer']['decision']==expected
    assert bool(searches)==(expected=='answer')
