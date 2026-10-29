# Paper Agent Harness

基于 nano-cc / learn-claude-code 设计思想的科研论文多轮检索问答 Agent。围绕指代缺失、证据不足和无依据作答，集成 Query 改写、混合检索、重排、证据反思、引用生成与会话记忆。

**当前状态：已实施 Harness 修订；本地机制回归与真实服务验收分开记录。** 效果评审为规则检查和模型辅助评审，未完成用户人工抽查；不宣称复现原截图的算法增益。

[最终验收收尾](reports/revision/CLOSURE.md) · [此前跟进](reports/revision/FOLLOWUP.md) · [此前修订](reports/revision/FINAL.md) · [修订前验收](reports/FINAL_ACCEPTANCE.md) · [安装与运行](docs/setup.md) · [架构](docs/architecture.md) · [评测协议](docs/evaluation.md)

## 核心实现

- 100 篇 QASPER 论文、1,996 个 chunks；段落合并与滑窗切块，保留论文、章节、段落和字符偏移。文本型 PDF 可导入，章节与阅读顺序需检查。
- BGE-M3 + FAISS 稠密检索、BM25、RRF 和 BGE Cross-Encoder 重排。
- Guardrails → NLU → Retrieve → Rerank → Reflect → Generate → Memory 七阶段编排，结构化意图/指代/改写、最多两轮检索、引用校验与受控拒答。
- 工作记忆、情景记忆、长期记忆：SQLite 权威快照/事件/确认声明，Redis TTL 与 Chroma 为可重建投影。论文检索独立使用 FAISS，普通 Claim 不写长期记忆。
- Turn/Stage Trace、按运行隔离的追踪与评分、有限重试、降级、稳定 turn_id 幂等提交、提交授权的归档可见性；CLI 支持 `/trace`、`/corpus`、`/memory`。

七阶段主链直接调用检索组件；独立 tool_loop 不参与主链的自主规划。引用结构校验不等于语义蕴含判断。

## 修订前历史实测结果

最终配置：`rerank + three_layer`，实际模型 `deepseek-v4.1-flash`，数据与源码版本见验收报告中的 freeze.json。

| 项目 | 结果 | 口径 |
|---|---:|---|
| 自动化回归 | 102 项通过 | 含崩溃恢复、真实 Chroma；73 条弃用警告 |
| Intent | 90/100 | 来源规则标签一致率 |
| Rewrite | 42/60，70.00% | 论文指代主视图，同模型辅助三条件判断 |
| Recall@5 | 66.26 / 73.94 / 74.24 / 77.61% | BM25 / Dense / Hybrid / Rerank；100 条已知论文问题 |
| Boundary | 49/55，89.09% | 决策一致率；拒答 P/R 为 76.92% / 100% |
| E2E | 120/120 执行成功 | 30 组四轮；不代表答案准确率 |
| 引用语义支持 | 340/377，90.19% | 同模型辅助评审，unknown 保留分母 |
| Memory 契约 | full / bounded 均 27/30 | 3 个九轮场景，不是通用记忆准确率 |

压缩组 token 本次增加 1.20%；Reflect 尚未证明稳定质量增益。检索提升的历史置信区间跨零。任务质量未知、误拒答与评审失败详见报告，不能只引用执行成功率。

## 快速开始

已有本机环境与资产：

```powershell
$py = '.\.venv-gpu\Scripts\python.exe'
& $py -m pytest -q
$env:MEMORY_BACKEND='three_layer'
$env:REDIS_URL='redis://localhost:16379/0'
& $py -m paper_agent.cli chat --index indexes/bge --mode rerank --encoder models/bge-m3 --reranker models/bge-reranker-v2-m3 --device cuda:0 --session demo
```

问答会使用 API 配额，需要 `.env`、模型、索引和真实 Redis。默认 CLI 为 SQLite 本地模式；完整模式需显式设置上述变量。新机器安装、固定模型下载、QASPER 与索引准备见 [安装文档](docs/setup.md)，不要覆盖现有凭据或重建有效资产。

## 目录

| 路径 | 用途 |
|---|---|
| `src/`、`tests/` | 实现与回归测试 |
| `configs/`、`scripts/` | 运行配置与数据/评测工具；见 [脚本导航](scripts/README.md) |
| `docs/` | 当前架构、组件、运行与评测文档 |
| `data/` | 评测集合与本地语料；正式主集 v0.6，旧版本保留溯源 |
| `reports/revision/CLOSURE.md` | 最新范围修复、真实集成、多轮对照与限制 |
| `reports/revision/FOLLOWUP.md` | 此前修复、验证与未完成项 |
| `reports/revision/FINAL.md` | 前一轮修订报告 |
| `reports/FINAL_ACCEPTANCE.md` | 修订前冻结验收，不代表新代码成绩 |
| `reports/final-acceptance/` | 修订前证据、逐轮结果、失败记录与成本账本 |
| `reports/history/` | 历史报告与开发实验，不能作为当前验收成绩 |

本机 `.venv*`、`models/`、`indexes/`、`runs/`、`.cache/`、`references/`、`tmp/` 为忽略的环境、资产或运行目录。Redis 二进制仍在 `tmp/memory-redis`，为避免破坏现有服务保持原位；无需为减少目录数迁移模型和环境。根目录旧计划已归入本地 `references/project-notes/`，导出物归入 `.cache/local-artifacts/`。

## 来源与许可

自有代码采用 [MIT](LICENSE)。上游固定版本、派生代码与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[upstream.lock](upstream.lock) 和 `licenses/`。模型、论文全文、索引、凭据及数据库运行目录不随仓库发布。
