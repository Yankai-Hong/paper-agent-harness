# 脚本导航

所有脚本从项目根目录运行。当前流程如下：

| 用途 | 脚本 |
|---|---|
| 模型/数据准备 | `download_models.py`、`prepare_qasper.py` |
| 检索四组 | `run_retrieval_suite.py`、`report_results.py` |
| 当前真实存储集成 | `check_memory_integration.py` |
| 当前完整验收 | `run_final_acceptance.py`，配置 `configs/final-acceptance.json` |
| Reflect诊断 | `reflect_diagnostic.py` |
| 当前证据汇总 | `summarize_final_acceptance.py`、`write_final_report.py`，固定读取当前证据目录 |
| 发布前只读核验 | `check_artifacts.py` |
| 检索历史复用准备 | `verify_retrieval_reuse.py`，一次性输出目录，已有结果不要重复执行 |

`phase_a_*`、`run_phase_a.py`、`report_phase_a.py`、旧 `acceptance/live` 工具保留用于历史复现，相关历史路径已转入 `reports/history/development/`。`run_dialogues.py`、`score_dialogues.py` 等仍被测试或组件流程使用，不因当前主入口改变而删除。

收费运行均应使用新输出目录并先确认有限请求预算。整理历史目录不会让旧模型结果自动变成当前源码结果。

## 多模态保留集

固定题集、许可证、范围及原页核查见 `data/eval/multimodal-holdout-v1/README.md`，冻结与结果见 `reports/revision/MULTIMODAL-HOLDOUT-20261031.md`。`freeze_multimodal_holdout.py` 独占创建源码哈希清单；`docling_ingest.py` 产生独立版本证据，`paper_agent.cli visual-collect`、`index --dense`、`visual-index`、`visual-semantic-index` 分别建立正文、CLIP 与 BGE 图注/表格索引。现有目录不能覆盖。

`run_multimodal_holdout.py` 需要预先准备三种索引、裁剪根和两个固定 CLIP 模型；它按题集原问题执行，输出目录必须不存在，每次真实请求在 `requests/` 预留并记录状态和可得用量，生成与自检响应只在本地通过敏感内容检查后保存，不上传外部遥测。它不是离线测试，调用模型可能产生费用；中断后须核对已有预留与账户情况，不自动续跑或将不明状态当作未发生。对保留集结果调参后必须重新准备独立保留集。

新版 `run_multimodal_v3_eval.py` 同时建立请求账本和声明审计；`join_multimodal_claim_audit.py` 只读合并引用坐标，`summarize_multimodal_v3_usage.py` 汇总可获得的真实用量。`score_multimodal_v2_retrieval.py` 和 `diagnose_holdout_v2_bm25.py` 是保留集执行后的只读分析；后者是重新运行的词法诊断，不得冒充首轮 BM25 轨迹。

`freeze_multimodal_holdout_v3.py` 保存本次题集与实现哈希，必须在首次解析前执行并使用新输出；`prepare_multimodal_holdout_v3.py` 验证原 PDF 并复现查询图。`run_bart_partial_images_v3.py` 仅在 ULMFiT 解析失败后运行 BART 单论文图搜图，不属于原定两论文无范围统计；详情见 `reports/revision/MULTIMODAL-V4-20261101.md`。

双论文开发复验使用 `run_multimodal_v3_eval.py` 的独立新账本及有限子查询预算；`summarize_multimodal_final_dev.py` 在运行后只读汇总固定目标、各通道候选和引用，不对语义自动打分。完整结果、未解决问题和复现参数见 `reports/revision/MULTIMODAL-FINAL-DEVELOPMENT-20261102.md`；不得将已参与开发的题集称为独立保留集。

`replay_context_repair.py` 对既存的本地生成声明使用其实际引用原文进行有界指代重放和逐声明核查；不执行新检索，不能当作新生成结果。新一轮代码快照、各过渡账本及 v3 全量和 v5 单题版本边界见 `reports/revision/MULTIMODAL-ACCEPTANCE-CLOSEOUT-20261102.md`。
# 图片查询开发评测

`eval_clip_image_text.py --help` 运行双论文、三张图片的 CLIP 检索评测，保留无范围 Top-5、原始图注目标、图搜图关联扩展和长片段诊断；目标只用于事后评分。`smoke_image_question.py --help` 以新目录和独立账本运行真实混合及纯图片问答，会向 `.env` 配置的远程模型发送图片并产生费用。测试不是独立保留集或人工金标，结果见 `reports/revision/MULTIMODAL-IMAGE-TEXT-20261106.md`。
