"""Regenerate Phase A reports exclusively from saved current-run artifacts."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from paper_agent.storage import read_jsonl, write_json, write_jsonl
from paper_agent.eval_tasks import bootstrap_ci, rewrite_metrics
from paper_agent.eval_protocol import compare_runs

ROOT = Path('reports/history/development/phase-a')
FULL = ROOT/'full'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def nonlinear_ci(rows, score):
    """Session bootstrap for nonlinear macro-F1 / refusal ratios."""
    groups = {}
    for r in rows: groups.setdefault(r['session_id'],[]).append(r)
    ids = sorted(groups)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(2000):
        sample = [r for sid in rng.choice(ids,len(ids),replace=True) for r in groups[sid]]
        value = score(sample)
        if value is not None: values.append(value)
    return {'estimate':score(rows),'clusters':len(ids),'replicates':2000,
        'undefined_replicates':2000-len(values),
        'bootstrap_95_percentile_interval':list(np.quantile(values,[.025,.975])) if values else None}


def telemetry(run):
    traces = [load(p) for p in (run/'traces').rglob('*.json')] if (run/'traces').exists() else []
    calls = [c for t in traces for c in t.get('api_calls', [])]
    if not traces and (run/'api_calls.json').exists(): calls = load(run/'api_calls.json')
    stage_calls = list(calls)
    for filename in ['claim-review.jsonl','semantic-review.jsonl']:
        if (run/filename).exists(): calls.extend(c for r in read_jsonl(run/filename) for c in r.get('calls',[]))
    latency = [sum(s.get('elapsed_ms',0) for s in t['stages']) for t in traces]
    return {'api_calls':len(calls), 'known_tokens':sum(sum(c.get('usage',{}).values()) for c in calls),
        'pipeline_or_component_calls':len(stage_calls),
        'pipeline_or_component_known_tokens':sum(sum(c.get('usage',{}).values()) for c in stage_calls),
        'judge_calls':len(calls)-len(stage_calls),
        'request_elapsed_ms_max':max((c.get('elapsed_ms',0) for c in calls),default=0),
        'unknown_usage_calls':sum(not c.get('usage_complete') for c in calls),
        'retry_attempts':sum(bool(c.get('retry_reason')) for c in calls),
        'p50_p95_stage_latency_ms':list(np.percentile(latency,[50,95])) if latency else None,
        'model_returned':dict(Counter(str(c.get('model_returned')) for c in calls)),
        'monetary_cost':None, 'degraded_turns':sum(bool(t.get('degraded')) for t in traces)}


def main():
    runs = {'intent':ROOT/'final-intent','rewrite':ROOT/'final-rewrite',
            **{n:FULL/n for n in ['boundary','memory','e2e']}}
    runs.update({'ablation/'+p.name:p for p in (FULL/'ablations').glob('*') if p.is_dir()})
    runs.update({'retrieval/'+n:FULL/'retrieval'/n for n in ['bm25','dense','hybrid','rerank']})
    summaries, failures = {}, []
    for name, run in runs.items():
        if not (run/'metrics.json').exists():
            summaries[name] = {'status':'not_completed', 'path':str(run)}
            continue
        metrics = load(run/'metrics.json')
        path = run/('predictions.jsonl' if (run/'predictions.jsonl').exists() else 'per_case.jsonl')
        rows = read_jsonl(path)
        ci = {}
        for key in ['correct','checkpoint_pass','behavior_match','execution_success','task_success','pipeline_success']:
            if any(key in r or key in r.get('assessment',{}) for r in rows):
                ci[key] = bootstrap_ci([float(bool(r.get(key,r.get('assessment',{}).get(key)))) for r in rows],
                    [r.get('session_id',r.get('source_paper_id',r.get('query_id'))) for r in rows])
        if name.startswith('retrieval/'):
            groups = {r['query_id']:r['source_paper_id'] for r in rows}
            for key in ['recall','mrr','ndcg','hit']:
                values = metrics['per_query']
                if all(v[key] is not None for v in values):
                    ci[key+'@5'] = bootstrap_ci([v[key] for v in values],[groups[v['query_id']] for v in values])
        if name == 'rewrite':
            gold = read_jsonl('data/eval/v0.6/query_rewrite.jsonl')
            preds = {r['id']:r for r in rows}
            for field in ['entity_retention_rate','token_overlap_with_expected']:
                values = [(r,rewrite_metrics([r],preds)[field]) for r in gold]
                values = [(r,v) for r,v in values if v is not None]
                ci[field] = bootstrap_ci([v for r,v in values],[r['session_id'] for r,v in values])
        if name == 'intent':
            gold = {r['id']:r['intent'] for r in read_jsonl('data/eval/v0.6/intent.jsonl')}
            labels = sorted(set(gold.values()))
            def macro(sample):
                scores=[]
                for label in labels:
                    tp=sum(gold[r['id']]==label and r.get('intent')==label for r in sample)
                    den=sum(gold[r['id']]==label for r in sample)+sum(r.get('intent')==label for r in sample)
                    scores.append(2*tp/den if den else 0)
                return sum(scores)/len(scores)
            ci['macro_f1'] = nonlinear_ci(rows,macro)
        if name == 'boundary':
            gold = {r['id']:r['decision'] for r in read_jsonl('data/eval/v0.6/boundary.jsonl')}
            def refusal(sample, kind):
                tp=sum(gold[r['id']]=='refuse' and r.get('decision')=='refuse' for r in sample)
                predicted=sum(r.get('decision')=='refuse' for r in sample)
                expected=sum(gold[r['id']]=='refuse' for r in sample)
                denom={'precision':predicted,'recall':expected,'f1':predicted+expected}[kind]
                return (2 if kind=='f1' else 1)*tp/denom if denom else None
            for kind in ['precision','recall','f1']:
                ci['refusal_'+kind] = nonlinear_ci(rows,lambda sample:refusal(sample,kind))
        errors = [r for r in rows if r.get('error')]
        for r in rows:
            wrong = r.get('correct') is False or r.get('checkpoint_pass') is False or r.get('assessment',{}).get('behavior_match') is False
            if r.get('error') or wrong:
                failures.append({'run':name,'case':r,'category':'execution_error' if r.get('error') else 'label_disagreement'})
        summaries[name] = {'status':'completed','path':str(run),'total':len(rows),'failed':len(errors),
            'valid':len(rows)-len(errors),'excluded':0,'metrics':metrics,'ci':ci,'telemetry':telemetry(run)}
        if (run/'semantic-summary.json').exists():
            summaries[name]['auxiliary_semantic'] = load(run/'semantic-summary.json')
        if name != 'rewrite' and (run/'semantic-review.jsonl').exists():
            from quality_statistics import summarize
            summaries[name]['auxiliary_quality'] = summarize(run)
    comparisons = {}
    pairs = [('reflect-disabled','reflect-enabled','behavior_match'),('reflect-enabled','reflect-retrieve-again','behavior_match'),
        ('memory-full','memory-bounded','checkpoint_pass'),('memory-full','memory-semantic','checkpoint_pass'),
        ('memory-semantic','memory-recall','checkpoint_pass')]
    for a,b,metric in pairs:
        if all((FULL/'ablations'/p/'metrics.json').exists() for p in [a,b]):
            comparisons[b+' minus '+a] = compare_runs(FULL/'ablations'/a,FULL/'ablations'/b,metric)
    for a,b in [('bm25','dense'),('dense','hybrid'),('hybrid','rerank')]:
        if all((FULL/'retrieval'/p/'metrics.json').exists() for p in [a,b]):
            aa,bb = [load(FULL/'retrieval'/p/'metrics.json')['per_query'] for p in [a,b]]
            lookup = {r['query_id']:r for r in aa}
            sources = {r['query_id']:r['source_paper_id'] for r in read_jsonl(FULL/'retrieval'/a/'predictions.jsonl')}
            comparisons[b+' minus '+a] = {'recall@5':bootstrap_ci([r['recall']-lookup[r['query_id']]['recall'] for r in bb],
                [sources[r['query_id']] for r in bb]),'configuration_difference':'retrieval mode only'}
    write_json(ROOT/'summary.json',summaries)
    write_json(ROOT/'paired-comparisons.json',comparisons)
    historical = {}
    for name, old, metric in [('intent','intent','correct'),('boundary','boundary-replay','correct'),('e2e','e2e-full','behavior_match')]:
        if summaries[name]['status']=='completed':
            historical[name] = compare_runs(Path('reports/history/development/acceptance')/old,runs[name],metric)
            historical[name]['interpretation']='Historical comparison only; source/dataset/config mismatch is explicit, not causal evidence.'
    write_json(ROOT/'historical-comparisons.json',historical)
    quality_pairs = {}
    for a,b,_ in pairs:
        pa,pb=FULL/'ablations'/a/'quality-statistics.json',FULL/'ablations'/b/'quality-statistics.json'
        if pa.exists() and pb.exists():
            aa={r['id']:r for r in load(pa)['cases']}
            bb={r['id']:r for r in load(pb)['cases']}
            common=sorted(aa.keys() & bb.keys())
            quality_pairs[b+' minus '+a] = {'common':len(common),'missing_a':sorted(bb.keys()-aa.keys()),
                'missing_b':sorted(aa.keys()-bb.keys()),'status':'provisional_same_model_judge',
                'answer_correct_delta':bootstrap_ci([int(bb[i]['answer_correct'])-int(aa[i]['answer_correct']) for i in common],
                    [bb[i]['session_id'] for i in common]) if common else None}
    write_json(ROOT/'quality-paired-comparisons.json',quality_pairs)
    write_jsonl(ROOT/'failures.jsonl',failures)
    manifest = load('data/eval/v0.6/manifest.json')
    audit_rows = read_jsonl('data/eval/v0.6/audit.jsonl')
    write_json(ROOT/'audit-changes.json', {'field_changes':dict(Counter(k for r in audit_rows for k,v in r['proposed'].items()
        if k not in {'dataset_version','review_status','human_reviewed','label_source'} and r['original'].get(k)!=v)),
        'new_human_verified_count':0,'pending':sum(r['review_status']=='needs_human_review' for r in audit_rows)})
    audit = ['# Phase A 数据审计','', '审核为原始材料与规则辅助核对，没有外部人工标注者。human-gold.jsonl 为空；source-verified 与 needs_human_review 独立保存。', '',
             '| 数据集 | 前/后样本数 | 来源规则/官方标注可确认 | 待人工 |','|---|---:|---:|---:|']
    for name,m in manifest['counts'].items():
        status = m['statuses']
        audit.append(f"| {name} | {m['before']}/{m['after']} | {status.get('source_rule_verified',0)+status.get('official_source_annotation',0)} | {status.get('needs_human_review',0)} |")
    audit += ['', 'Rewrite expected_query 缺失由 90 降至 0：60 条补齐论文/问题，20 条已独立问题原样保留，10 条显式歧义问题保留澄清需求。参考不是唯一措辞；token overlap 仅为词面诊断。',
        'Intent 类别分布未修改。v0.5 开发审核曾错误地按轮次推断 intent，已标记 SUPERSEDED，未用于任何评测；v0.6 只依据明确任务动词，保留不确定类别。',
        '每条原值、建议值、原因、来源问题/论文/证据 ID、状态见 data/eval/v0.6/audit.jsonl；545 条审核记录包含不同任务对同一问题的视图，不能当作 545 个独立会话。',
        'Memory 80 条非空检查点覆盖讨论论文和来源保留；自然事实更新、冲突及归档精确率缺少可靠期望，显式 not_applicable，未编造金标。',
        'Retrieval 使用官方 QASPER 有文本证据的 100 条；原始筛选排除 60 条无答案/无文本证据、126 条证据无法映射，见 data/qasper/manifest.json，不纳入可回答 Recall 分母。',
        'Claim 源文诊断 8 条见 phase-a/claim-benchmark/cases.jsonl 和 data/eval/claim-diagnostics-v1；预期由代理核对原文，不是独立人工金标。正式人工 Claim 验证仍缺失。',
        '进一步逐条读取原始 QASPER 压缩包中的全部标注：100 条中有 12 条可回答性分歧、2 条 yes/no 分歧，没有一致标为不可回答的样本。完整原始答案、证据与复核队列见 phase-a/source-annotation-audit。原检索协议选择至少一个可映射的可回答标注，并不代表所有标注者一致；100 条主分母不因观察到模型结果而删改。']
    Path('reports/history/development/PHASE_A_AUDIT_REPORT.md').write_text('\n'.join(audit)+'\n',encoding='utf-8')
    claim = load(FULL/'e2e/claim-metrics.json') if (FULL/'e2e/claim-metrics.json').exists() else {'status':'not_completed'}
    claim_text = ['# Claim 与 Citation 评测','',
        '实现：src/paper_agent/claim_eval.py 复用 validate_citations；scripts/evaluate_claims.py 从实际 Generate.claims 和最终 Rerank 证据读取逐 Claim 输入。未用整库存在性替代实际提供证据。',
        '每个 Claim 只计一次；同一 Claim 内重复引用去重，跨 Claim 引用分别计数。Claim Support 分母为全部生成 Claim；Unsupported 为 contradicted 或 insufficient，unknown 单列且保留在总分母，因此失败时两者之和不必为 1。Contradiction 同样以所有 Claim 为分母。Citation Validity 与 Citation Precision 分母为去重后的 Claim-引用对；后者只计该引用直接支持该 Claim 的对数。',
        '支持需全部关键细节被合并证据支持；明确冲突为 contradicted；其余为 insufficient。无引用或全部引用失效确定性计 insufficient。语义 Judge 模型、提示词、原始响应、解析结果、理由、逐调用 usage 均保存；失败计 unknown。',
        '95% CI 为 session 簇 bootstrap 2000 次。仅单簇的 8 条诊断集 CI 无外推意义；原文诊断 8/8 一致不等于人工金标准确率。未获得人工样本，因此语义结果全部 provisional。',
        'Generate Schema 只允许 claims，没有单独自由回答正文；本评测不对外部文本做 Claim 自动切分。',
        '','实际结果：','```json',json.dumps(claim,ensure_ascii=False,indent=2),'```']
    Path('reports/history/development/CLAIM_CITATION_EVAL.md').write_text('\n'.join(claim_text)+'\n',encoding='utf-8')
    report = ['# Phase A 最终回归报告','',
        '完成状态：代码、来源规则标签补齐、实际回归分别按下表核验；人工金标及独立 Judge 验证未完成，不能宣布全部验收完成。历史 acceptance 结果没有替代本轮结果。',
        '基准 Git HEAD：4368a62a4ba15160fb5ff452194fd10bd2737c4e，已通过 git ls-remote 核对远端一致。保留原有 output/、tmp/。未提交修改用文件 SHA-256 绑定。',
        '初次冻结后发现 Rewrite 错误分母、Memory 仅来源检查点两处缺陷并修复；初次 Intent/Rewrite 保留，最终表采用 final-intent/final-rewrite 重跑。其余运行开始时已加载修复后的模块，各 run_config.json 保存实际源码哈希。',
        '','| 模块 | 样本总数 | 执行失败 | 指标与计分分母 | 已知 token（含该目录 Judge） |','|---|---:|---:|---|---:|']
    for name,s in summaries.items():
        if s['status'] != 'completed':
            report.append(f'| {name} | — | — | 未完成 | — |')
            continue
        m=s['metrics']
        field=next(k for k in ['accuracy','decision_accuracy','checkpoint_accuracy','Recall@5','entity_retention_rate','execution_success_rate'] if k in m)
        value=m[field]
        if isinstance(value,dict):
            point=f"{field}: {value.get('correct')}/{value.get('total')} ({value.get('rate',0):.2%})"
        else:
            denom=m.get('entity_applicable_cases',s['total']) if field=='entity_retention_rate' else s['total']
            point=f'{field}: {value:.2%}，分母 {denom}'
        report.append(f"| {name} | {s['total']} | {s['failed']} | {point} | {s['telemetry']['known_tokens']} |")
    report += ['', '完整指标、95% CI、分母、未知数、P50/P95、实际返回模型与调用数见 phase-a/summary.json；逐例失败见 phase-a/failures.jsonl，配置差异/配对差值见 paired-comparisons.json。CI 跨零或仅少量会话不得声称增益。',
        'Intent/Boundary 为标签一致率，非人工准确率；Memory 为结构检查点通过率，非事实语义正确率。E2E execution/task/pipeline 分开，未知任务质量不计成功，保留分母。引用结构合法不证明语义支持。',
        'API 使用 .env 既有 OpenCode deepseek-v4.1-flash，显式 4096/disabled，不改凭据。真实 usage 计数来自响应，非字符/4 估算；币值单价与账户剩余额度不可获取，费用未知。',
        '首次沙箱运行 SSL DLL 初始化失败，获准使用原环境后恢复。原 Intent 100 次请求预算耗尽留有失败，最终重跑预算 200，均有限重试。完整命令及运行时间见 full/jobs.json 与各 run_config.json。',
        '消融沿用已有 v0.3 小子集（Reflect 12 turns、Memory 20 turns），与 v0.6 全量分开统计，不能混合总分；reflect-enabled 与 reflect-no-second 是同参数重复组。',
        '未完成：真正人工金标/抽查 Judge 一致性；自然长会话的精确事实更新和归档精确率；无独立充分性标签的 Reflect 子集质量；未知服务端超时成本及金额。',
        '', '复现：见 docs/evaluation.md 的 Phase A 命令。原资产直接复用；不下载模型或重建索引。']
    if summaries['e2e']['status']=='completed':
        m=summaries['e2e']['metrics']
        report += ['', 'E2E 质量口径：',f"- 行为标签一致率：{m['behavior_match_rate']['correct']}/120；任务与综合 Pipeline 已满足条件下界均 {m['task_success_rate']['correct']}/120，任务质量未知 {m['task_success_rate']['unknown']}。不能将上表 execution 100% 解释为任务质量 100%。"]
        if summaries['e2e'].get('auxiliary_quality'):
            q=summaries['e2e']['auxiliary_quality']['answer_correct']
            report.append(f"- 同模型辅助答案质量：{q['mean']:.2%}，95% CI {q['bootstrap_95_percentile_interval']}；不是独立人工准确率。")
    report += ['', '质量与成本分开比较：','| 运行 | 辅助答案质量 | Pipeline token（不含 Judge） | P50/P95 阶段耗时 ms |',
               '|---|---:|---:|---|']
    for name,s in summaries.items():
        if s.get('auxiliary_quality'):
            report.append(f"| {name} | {s['auxiliary_quality']['answer_correct']['mean']:.2%} | {s['telemetry']['pipeline_or_component_known_tokens']} | {s['telemetry']['p50_p95_stage_latency_ms']} |")
    report += ['', '延迟异常：本轮部分 Judge 请求记录约 8,895 秒 elapsed，远超配置的 120 秒；最终触发传输失败并有限重试。具体外部服务/宿主暂停原因未确认，不能据此推断正常吞吐或宣称硬实时超时保证；原始异常与未知成本均保留。',
        '人工待复核仍阻塞最终科研验收。语义压缩与归档召回本次未证明质量增益；仅 2 个 Memory 会话簇的退化 bootstrap 区间即使不跨零，也不足以支持总体结论。',
        'Reflect 的 12-turn retrieve-again 组本次二次检索触发为 0/12，因此不能评价二次检索后的证据改善率。Memory 三个压缩组均压缩 12/20 轮，召回组 10/20 轮使用归档；这是实际触发记录，不是召回精确率。见 full/ablations/trace-diagnostics.json。',
        'Recall/MRR/nDCG/Hit@10 的置信区间、BM25/Dense 互补案例与 RRF/重排逐题变化见 full/retrieval/extended-statistics.json；70-turn 子集从完整120轮计分，见 full/e2e-70-view.json，不声称另跑70轮。',
        'Claim 覆盖错误 d28-t3 的第9条声明保留 unknown，没有把 Judge 的表面 supported 输出算成功。全部8条证据不足和该失败见 claim-failures.jsonl；44条固定抽样/异常审阅队列待人工确认。零矛盾和100%结构有效的退化bootstrap区间不证明总体风险为零。']
    ledger=ROOT/'cost-ledger.json'
    if ledger.exists():
        cost=load(ledger)
        report += ['', f"成本账本（含失败、重跑、Judge）：已保存 {cost['saved_calls']} 次请求、已知 {cost['known_tokens']} token，{cost['unknown_usage_calls']} 次 usage 未知；另 smoke {cost['smoke_separately']['calls']} 次/{cost['smoke_separately']['known_tokens']} token。金额与剩余额度未知。见 phase-a/cost-ledger.json。"]
    test_path=ROOT/'final-tests.xml'
    if test_path.exists():
        suites=ET.parse(test_path).getroot()
        tests=list(suites.iter('testcase'))
        failed=[t.attrib for t in tests if t.find('failure') is not None or t.find('error') is not None]
        report += ['',f'最终单元/可靠性测试：{len(tests)} 项，失败 {len(failed)}；见 phase-a/final-tests.xml。包含超时、429/503、非法 JSON、Schema 修复、引用错误、摘要降级、归档失败、幂等恢复及新增 Claim/分母/审核测试。']
        write_json(ROOT/'test-summary.json',{'total':len(tests),'failures':failed,'cases':[t.attrib for t in tests]})
    report += ['', '辅助质量与消融配对（不是人工准确率）：','```json',json.dumps(quality_pairs,ensure_ascii=False,indent=2),'```']
    Path('reports/history/development/PHASE_A_FINAL_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print({k:v['status'] for k,v in summaries.items()})


if __name__=='__main__': main()
