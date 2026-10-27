"""Render the concise engineering acceptance report from recomputed local evidence."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from summarize_final_acceptance import percent

root=Path('reports/final-acceptance')
s=json.loads((root/'summary.json').read_text(encoding='utf-8'))
diagnostic_name='reflect-diagnostic-v2' if (root/'reflect-diagnostic-v2/summary.json').exists() else 'reflect-diagnostic'
d=json.loads((root/diagnostic_name/'summary.json').read_text(encoding='utf-8'))
diagnostic_cost={'requests':d['requests'],'known_tokens':0,'unknown_usage_requests':0}
for path in (root/diagnostic_name/'requests').glob('*.json'):
    usage=json.loads(path.read_text(encoding='utf-8')).get('usage') or {}
    if isinstance(usage.get('prompt_tokens'),int) and isinstance(usage.get('completion_tokens'),int):
        diagnostic_cost['known_tokens']+=usage['prompt_tokens']+usage['completion_tokens']
    else: diagnostic_cost['unknown_usage_requests']+=1
blocked=d.get('status','').startswith('blocked')
tests=ET.parse(root/'final-v2-tests.xml').getroot().find('testsuite').attrib
b=s['boundary']['metrics']
c=s['citations']
e=s['e2e120']
seventy=s['e2e70']['metrics']
q=s['auxiliary_quality']
cost=s['costs']
token=s['memory_token_comparison']
lines=[
'# Paper Agent Harness 最终工程验收',
'',
'本报告面向本地工程展示与实习简历，不是论文级算法验证。所有模型辅助评审均未获用户人工抽查；同模型 Judge 不作为人工准确率。',
'',
f"基准 Git HEAD：`{s['configuration']['git_head']}`。实验开始时工作区干净；该 SHA 是实验基准，最终修复与发布整理由后续 Git 提交记录。最终运行源码逐文件 SHA-256、配置、模型、数据和索引见 [freeze.json](final-acceptance/live-v2/freeze.json)。文档编辑发生在冻结后，不影响运行代码。",
'',
'完整配置：BGE-M3（固定 revision）+ FAISS IndexFlatIP，BM25，两路各30候选，RRF k=60，Cross-Encoder 前30重排、返回5；七阶段 Pipeline，three_layer（真实 Redis 7.0.15 / SQLite / Chroma 0.6.3），bounded 工作记忆。LLM 实际请求/返回模型 deepseek-v4.1-flash，4096输出上限、thinking disabled。固定回答器集成与真实LLM结果分开。',
'',
'## 修复与回归',
'',
'| 问题 | 修改位置 | 行为变化与回归 |',
'|---|---|---|',
'| 未提交压缩归档可召回 | memory.py、pipeline.py、compaction.py、memory_manager.py | 新归档 ID 必须存在于权威提交快照。legacy 以原子 state 文件替换为提交点；three_layer 以 SQLite 事务为提交点。归档原文先写、状态后裁剪；不靠异常删文件。测试覆盖两模式×两策略提交失败重启、正常召回、同ID重放及两个实际子进程 os._exit 崩溃。 |',
'| 跨论文相同文本误判冲突 | memory_stores.py / FactStore.put_fact | 按来源、论文范围和 canonical attribute/key 比较。不同论文/属性共存；同属性需显式替代，保留 superseded 与 conflict。测试包含来源隔离、多论文过滤和冲突非权威召回。 |',
'| 新 Chroma 目录并行初始化失败 | memory_stores.py / ChromaEpisodes.collection | 同进程首次初始化加锁，避免读取未迁移完成的 tenants 表；真实3线程/3实例回归。不是生产多进程并发承诺。 |',
'| 配置归属与实际链路不清 | eval_cli.py、check_memory_integration.py、run_final_acceptance.py、configs/final-acceptance.json | 记录真实 backend/mode、源码和数据哈希、调用预算；固定集成由 hybrid 升至真实 rerank，新增部分 Chroma 删除恢复。 |',
'',
f"最终测试：**{tests['tests']} 项，失败 {tests['failures']}，错误 {tests['errors']}，跳过 {tests['skipped']}**；73条 Chroma/Pydantic 弃用警告。证据：[JUnit](final-acceptance/final-v2-tests.xml)、[日志](final-acceptance/final-v2-tests.log)。新提交/崩溃回归见 tests/test_commit_boundaries.py。",
'',
('最终源码的真实固定回答器集成重新通过：6轮、重排、Redis TTL过期恢复、幂等重放、Chroma部分缺失恢复、压缩和召回，详情见 [集成结果](final-acceptance/integration-v2/result.json)。用户授权仅重启Ubuntu后恢复既有Redis；没有重启Windows或Docker。该结果不计入LLM答案准确率。' if (root/'integration-v2/result.json').exists() and json.loads((root/'integration-v2/result.json').read_text())['status']=='passed' else '真实固定回答器六轮集成见 [原集成结果](final-acceptance/integration/result.json)，执行于Chroma初始化锁修复前，未冒充最终源码再次执行。'),
'',
'首次真实并行运行发现 Chroma 初始化竞争：保留了失败和中断产物，未将部分会话补齐后冒充完整结果。修复后在 live-v2 重跑全部120轮。Intent/Rewrite 已完整执行且不依赖存储，复用 live 中的本次 NLU-only 结果，各目录含 reuse-provenance.json；费用不重复计算。',
'',
'## 最终评测总表',
'',
'下表 F=上述完整配置；N=同模型 NLU-only；R=独立检索四组。所有路径相对于 reports/final-acceptance。',
'',
'| 指标 | 数据/分母 | 运行配置 | 结果 | 证据路径 | 局限 |',
'|---|---|---|---|---|---|',
f"| Intent | v0.6 / 100 | N，本次已完成结果复用 | {percent(s['intent']['metrics']['accuracy'])} | live-v2/intent/metrics.json | 来源规则标签，37条仍待人工，不改标签追分 |",
f"| Rewrite | v0.6 / 90；论文指代主视图60 | N + 同模型Judge | 主视图三条件同时通过 {percent(s['rewrite']['paper_followup_all_three'])} | summary.json / rewrite；live-v2/rewrite/review.jsonl | 指代、原意约束、独立检索分别判断；非人工准确率 |",
f"| Rewrite辅助 | 实体适用60 | N | 实体保持 {s['rewrite']['metrics']['entity_retention_rate']:.2%} | live-v2/rewrite/metrics.json | 不是改写准确率；全部90评审另存 |",
f"| Recall@5 | 同100条QASPER官方证据问题 | R，known-paper | BM25 66.26%；Dense 73.94%；Hybrid 74.24%；Rerank 77.61% | retrieval-reused/summary.json、reuse-verification.json | 2026-10-26独立重跑复用；不是全库无论文提示检索、也不是答案正确率 |",
f"| Boundary | v0.6 / 55 | F；其中{s['boundary']['same_run_e2e_reused']}条精确复用本次完整E2E历史 | 决策 {percent(b['decision_accuracy'])} | live-v2/boundary/metrics.json | 失败保留分母；仅20条来源规则确认 |",
f"| 拒答 | 同55条 | F | precision {percent(b['refusal_precision'])}；recall {percent(b['refusal_recall'])}；误拒答 {percent(b['false_refusal_rate'])} | 同上 | 误拒答分母仅应回答样本；越界/歧义/证据不足分列summary |",
f"| 引用ID/来源合法 | {c['id_source_validity']['denominator']}条最终claim-引用对 | F + 规则校验 | {percent(c['id_source_validity'])} | claims.jsonl | 仅最终通过校验的输出；不代表原始生成从未出现非法引用，更不等于语义支持 |",
f"| 引用语义支持 | 同引用对；{c['claims']}条Claims | F + 同模型Judge | 引用 {percent(c['semantic_citation_support'])}；Claim {percent(c['claim_support'])} | claims.jsonl、live-v2/e2e/review.jsonl | unsupported/contradicted/unknown见下；未独立人工验证 |",
f"| E2E execution / behavior | 30×4=120轮 | F，完整历史执行 | execution {percent(e['execution_success_rate'])}；behavior {percent(e['behavior_match_rate'])} | live-v2/e2e/metrics.json | behavior仅决策标签一致，不等于任务质量 |",
f"| Task / Pipeline | 120轮 | F，未提供人工质量 | 下界 {percent(e['task_success_rate'])} / {percent(e['pipeline_success_rate'])}；unknown {e['task_success_rate']['unknown']} | 同上 | 未知不计成功；同模型辅助任务正确另列 {percent(q['e2e']['task_correct'])} |",
f"| 固定70-turn视图 | 从完整120轮取70 | F，相同运行 | execution {percent(seventy['execution_success_rate'])}；behavior {percent(seventy['behavior_match_rate'])}；pipeline下界 {percent(seventy['pipeline_success_rate'])} | summary.json / e2e70 | 未跳过依赖历史；unknown {seventy['task_success_rate']['unknown']} |",
f"| 压缩后Memory一致性 | 3场景×9轮×2设置；每组30项具体契约 | F，full vs bounded | full {percent(s['memory']['memory-full']['checks'])}；bounded {percent(s['memory']['memory-bounded']['checks'])} | memory-checks.jsonl | 显式约束/值/版本/冲突/未解决事件，不是任意自然对话准确率 |",
f"| Reflect自然消融 | 同固定12轮×3；另6诊断×3 | F，仅改变Reflect/检索尝试上限 | 二次检索 off/on/second = {s['traces']['reflect-off']['second_retrieval']}/{s['traces']['reflect-on']['second_retrieval']}/{s['traces']['reflect-second']['second_retrieval']}；诊断见下 | live-v2/reflect-*/；{diagnostic_name}/ | 历史v0.3子集单列，不与v0.6合并；强制分支不是模型效果 |",
f"| Token与质量 | 相同27轮×2、同模型 | F，full vs bounded | {cost['memory-full']['known_tokens']:,} → {cost['memory-bounded']['known_tokens']:,} 已知token；辅助任务质量 {percent(q['memory-full']['task_correct'])} → {percent(q['memory-bounded']['task_correct'])} | summary.json / memory_token_comparison；requests.jsonl | 所有实际NLU/Reflect/Generate/重试计入；摘要调用0（确定性压缩）；Judge另计；未知usage不算0 |",
'',
'## 标签、错误与运行证据',
'',
f"Intent各类数量：`{json.dumps(s['intent']['class_counts'],ensure_ascii=False)}`。10个不一致中8个是私有未公开实验：标签 fact、模型 out_of_scope；两种标签存在分类边界歧义，仍按原规则计错。另有明确“检索原文”被判fact，以及开放解释型问题fact/explain分歧。逐条错误见 summary.json / intent.errors。",
f"Rewrite 60条论文后续问题的指代正确、含义约束保持、可独立检索分别为：{'; '.join(k+' '+percent(v) for k,v in s['rewrite']['paper_followup_subset'].items())}。90条全量保留10条应澄清问题及20条非论文检索问题；不为提高主指标删除失败。",
f"Claim verdict统计：`{json.dumps(c['verdict_counts'])}`。insufficient即未获证据支持，contradicted为与证据矛盾，unknown为无法判断/评审失败。典型案例见 review-package.jsonl 和完整 claims.jsonl，不把unknown当supported。",
f"实际E2E trace：`{json.dumps(s['traces']['e2e'])}`。短四轮会话不必触发压缩；长会话bounded实际压缩{s['traces']['memory-bounded']['compressed_turns']}/27轮，召回{s['traces']['memory-bounded']['recall_turns']}/27轮，archive来源{s['traces']['memory-bounded']['archive_record_turns']}/27轮。Memory契约失败（若有）保留在summary，不用单纯论文ID命中代替事实一致性。",
'',
'Reflect额外诊断在输出前固定为前三条官方可回答问题，加同论文未公开原始日志SHA256请求，共6例。正确拒答分母固定为3，不视为完整算法准确率：',
'',
'| 设置 | 诊断决策一致/6 | 正确拒答/3 | 实际二次检索 | 执行失败 |',
'|---|---|---|---|---|']
for name,r in d['natural'].items(): lines.append(f"| {name} | {r['decision_match']}/6 | {r['correct_refusal']}/3 | {r['second_retrieval']}/6 | {r['failures']} |")
lines += [
'',
('补充诊断的18个案例全部在Redis读取阶段失败，API调用0；上表零值是执行失败，不是模型效果。WSL/Ubuntu跨夜后无响应、16379端口不可用，重启Ubuntu需避免中断用户其他进程，已询问但尚未取得确认。真实强制分支同样未执行成功；已有pytest固定回答器二次检索上限测试通过（BM25，机制测试），不冒充完整模式自然效果。' if blocked else '强制分支使用固定回答器和真实检索/存储，验证最多二次检索后拒答，通过；API调用0。自然分支若未触发或未提高质量，就只能说明该固定样本上的结果，不能据此宣布二次检索普遍有效。'),
'',
'12-turn自然子集同模型辅助任务质量：'+ '；'.join(name+' '+percent(q[name]['task_correct']) for name in ['reflect-off','reflect-on','reflect-second'])+'。该子集包含可回答、跨论文及越界场景，并非全部证据不足。',
('补充诊断不能直接解读为质量增益：off组两条insufficient样本虽输出decision=answer，但正文明确声明没有校验和证据，没有编造SHA256。on组positive-0误拒答；second组positive-0并未触发二次检索，其组间决策差异不能归因于二次检索。实际触发的positive-1前后官方段落证据覆盖代理指标均为1.0，没有证明检索召回提升。细节见 reflect-diagnostic-v2/results.jsonl 与 secondary-retrieval-evidence.json。' if not blocked else '补充诊断尚无可解释的自然模型结果。'),
'',
f"复核包 {s['review_package']['cases']} 例：固定正常样本与问题富集样本分别标识，含Rewrite和长Memory契约；完整失败/unknown列表另存 [failures.jsonl](final-acceptance/failures.jsonl)。抽样不是总体准确率。当前为Agent/模型辅助复核，human_reviewed全部false，用户人工抽查待完成。",
'Agent逐条读取证据的6例定性复核见 [AGENT_REVIEW_NOTES.md](final-acceptance/AGENT_REVIEW_NOTES.md)：发现地域范围遗漏、部分证据能否替代完整比较的口径分歧、Judge索引覆盖失败等。未据此改标签或覆盖模型分数。',
'Memory最后一轮询问历史值时，模型有时按论文助手边界拒答；状态检索成功不代表已实现通用个人记忆问答。辅助Judge也可能将此拒答判正确，因此不能用其质量分数替代具体记忆契约检查。E2E与bounded各有1轮citation_validation_failed降级拒答，已留在trace中，未隐藏。',
'',
'成本账本（包括中断、重跑与评审）：',
'',
'| 组别 | 请求 | 已知token | usage未知请求 |',
'|---|---:|---:|---:|',
f"| 初次中断运行（含已复用NLU） | {s['initial_run_costs']['requests']} | {s['initial_run_costs']['known_tokens']} | {s['initial_run_costs']['unknown_usage_requests']} |",
* [f"| {name} | {r['requests']} | {r['known_tokens']} | {r['unknown_usage_requests']} |" for name,r in cost.items()],
f"| 补充真实诊断 | {diagnostic_cost['requests']} | {diagnostic_cost['known_tokens']} | {diagnostic_cost['unknown_usage_requests']} |",
f"| 合计 | {s['initial_run_costs']['requests']+sum(r['requests'] for r in cost.values())+d['requests']} | {s['initial_run_costs']['known_tokens']+sum(r['known_tokens'] for r in cost.values())+diagnostic_cost['known_tokens']} | {s['initial_run_costs']['unknown_usage_requests']+sum(r['unknown_usage_requests'] for r in cost.values())+diagnostic_cost['unknown_usage_requests']} |",
'',
f"预算总上限1800次（初次423、新运行1233、额外诊断144），本次未超限。bounded相对full已知token变化 {token['known_token_delta']:+,}（节省率 {token['savings_rate']:.2%}），没有节省；质量仅为同模型辅助判分，22/27与23/27不足以声称总体增益。金额/实际账户扣费无法核实，不推测花费。",
'',
'## 截图评测覆盖',
'',
'| 类别 | 状态 | 边界 |',
'|---|---|---|',
'| 30组四轮/120 turns | 当前完整配置已执行 | 全部保留，失败计入 |',
'| 100 Intent | 本次已执行，NLU结果复用 | 不是三层存储测试 |',
'| 90 Rewrite | 本次执行+同模型辅助评审 | 60条论文指代主视图，全部90逐例保留 |',
'| 100检索/四组Recall@5 | 复用2026-10-26独立重跑 | 源码/实际权重/tokenizer/索引/数据校验，无重新计算冒称 |',
'| 55 Boundary/拒答 | 当前配置已执行或精确复用本次E2E | 完整历史，不注入虚构助手回答 |',
'| 70-turn E2E | 已从完整120轮计分 | 不是独立跳轮运行 |',
'| 引用 | 当前Claims规则校验+同模型语义Judge | 人工语义准确率无法验证 |',
'| Pipeline Success | execution/behavior/task/pipeline分别报告 | 未审阅质量列unknown；模型辅助另列 |',
'| Memory | 3×9轮full/bounded；真实三层集成 | 显式状态契约，非通用语义记忆能力 |',
('| Reflect消融 | 当前12轮×3已执行；额外六诊断×3执行失败 | 主组实际二次检索1/12；补充真实强制分支受Redis阻塞，不能用Mock替代 |' if blocked else '| Reflect消融 | 12轮×3 + 六诊断×3 + 强制分支 | 无提升亦保留；自然与故障机制分开 |'),
'| Token | 实际usage、重试、Judge分别记账 | 无usage列unknown，金额无法核实 |',
'',
'## 验收与限制',
'',
'已完成：归档提交漏洞修复、跨论文冲突修复、新发现的Chroma初始化竞争修复；真实完整链路、恢复/回放测试；所列评测类别均有明确结果及产物。',
'',
'可接受限制：旧归档若无可核对提交账本会保留但不自动召回；属性同义词/通用语义矛盾不自动裁决；逐字论文句子以外的事实保守存为conflict；user/conversation仍有保守词项相似规则。SQLite摘要持写锁、长期账本/归档ID增长、归档线性扫描、有限重试非绝对墙钟超时均保留。Chroma弃用警告未隐藏。不承诺生产并发、通用记忆或绝对无幻觉。',
'',
'展示口径：适合展示可审计的研究型Agent工程，不可宣称论文级总体增益或人工答案准确率。需查看失败样例演示边界；缺少用户人工抽查不阻止工程展示，但必须保留“模型辅助/待人工抽查”。提交与发布状态以 Git 历史和远端分支为准；此处保留实验运行时的版本依据。',
('仍阻塞现场实时展示的事项：当前Ubuntu/Redis不可达。代码与已经保存的完整运行可审阅，不能声称此刻服务在线。最少需要用户允许或自行重启Ubuntu并启动既有16379 Redis，再运行真实集成；可选补跑既有固定六例诊断，无需修改数据或下载模型。' if blocked else '当前无阻塞现场展示的已知基础设施问题；仍需使用现有资产和真实Redis启动。'),
('此前跨夜WSL失联导致的18个诊断执行失败保留在 reflect-diagnostic/，没有收费请求；恢复后的完整固定子集另存 reflect-diagnostic-v2/，不与失败批次混合或覆盖。' if not blocked else '环境阻塞记录和全部失败已保留，未用Mock替代。'),
'',
'## 复现与发布',
'',
'先安装现有依赖/启动Redis；模型、数据、索引未随仓库发布，按README获取固定资产。已有环境不下载或重建。完整命令与预算协议见 [docs/evaluation.md](../docs/evaluation.md)。',
'',
'```powershell',
"$py = '.\\.venv-gpu\\Scripts\\python.exe'",
"$env:HF_HUB_OFFLINE='1'",
"$env:TRANSFORMERS_OFFLINE='1'",
'& $py -m pytest -q --junitxml=reports/reproduced-tests.xml',
'& $py scripts/check_memory_integration.py --redis-url redis://localhost:16379/0 --output reports/reproduced-integration',
'& $py scripts/run_final_acceptance.py --output reports/reproduced-live --config configs/final-acceptance.json',
'& $py scripts/run_final_acceptance.py --output reports/reproduced-live --config configs/final-acceptance.json --phase judge',
'& $py scripts/reflect_diagnostic.py --output reports/reproduced-reflect-diagnostic',
'# 单次交互演示（真实Redis已运行）',
"$env:MEMORY_BACKEND='three_layer'",
"$env:REDIS_URL='redis://localhost:16379/0'",
"& $py -m paper_agent.cli chat --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session demo-final",
'```',
'',
'实验结束时的变更快照见 [changed-files.txt](final-acceptance/changed-files.txt)。原始运行保留，Chroma/SQLite运行目录、模型/索引/凭据不纳入提交；memory-checks.jsonl导出具体契约证据。发布顺序：审阅diff与失败案例 → git diff --check及必要测试 → 手动git add所列源码/测试/配置/文档/证据 → 检查暂存区无.env、权重和原始私有数据 → 用户自行commit → 用户自行push/发布。该清单为实验结束时快照，后续发布整理以 Git 提交为准。',
'',
'## 简历可用描述',
'',
f"实现论文检索 Agent Harness，集成 BGE-M3/FAISS、BM25/RRF 与 Cross-Encoder，构建七阶段问答流程及 Redis–SQLite–Chroma 三层记忆；修复压缩归档提交隔离、跨论文事实冲突与并行初始化问题，通过 {tests['tests']} 项回归测试；完成30组120轮对话、100条Intent、55条Boundary及压缩/Reflect消融评测，在100条QASPER已知论文检索上 Recall@5 达77.61%，提供逐例trace、来源校验与模型辅助质量复核。",
'',
'该描述不将检索Recall写成答案正确率，不将同模型Judge写成人工准确率，也不承诺压缩一定提高质量或节省token。',
]
Path('reports/FINAL_ACCEPTANCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote reports/FINAL_ACCEPTANCE.md')
