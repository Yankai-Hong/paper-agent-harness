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
