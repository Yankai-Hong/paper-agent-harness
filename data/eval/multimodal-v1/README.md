# 多模态验收题集 v1（待独立人工复核）

`cases.json` 在多模态实验前固定：10 个问答题（3 正文、3 架构、2 图表/表格、1 正文与图、1 证据不足），3 个图搜图题。页码为从 1 开始的 PDF 物理页码。助手已查看对应原页并核对预期答案；这不是独立人工金标。无依据不猜测，`u01` 是拒答探针，并非证明外部不存在某个权重文件。所有问题与预期答案均不依赖生成模型。小样本不能代表总体效果。

论文为 QASPER 语料中对应的 arXiv 公开版本，实际下载版本由哈希固定。arXiv 标示的许可是授予 arXiv 的非独占分发许可，不能因此假设第三方有再分发权；原 PDF、原页面和裁剪仅存于已忽略的 `data/raw/`，不应提交到公开仓库。任务中仅授权向现有 OpenCode Go 服务发送必要图表和上下文，不授权向外部遥测发送内容。QASPER 数据集的 CC BY 4.0 不自动覆盖各论文版权。

| 论文 | 版本与来源 | PDF SHA-256 |
|---|---|---|
| Fine-tune BERT for Extractive Summarization | [1903.10318v2](https://arxiv.org/abs/1903.10318v2) | `c1fa1e935ff036918922a7c63ca59a55cd0da2d6a9d6f861cb41a879df636e1b` |
| Hint-Based Training for Non-Autoregressive Machine Translation | [1909.06708v1](https://arxiv.org/abs/1909.06708v1) | `f3877e51d7e117ce4b3f46faacd60e974e2bfa64d958ecd6c710a5dc917e551e` |
| Improving Fine-grained Entity Typing with Entity Linking | [1909.12079v1](https://arxiv.org/abs/1909.12079v1) | `33826fdaafbc2b5e25e956dd0439261373a1e87f6a87313a41e81dcbb303fbcb` |
| TENER | [1911.04474v3](https://arxiv.org/abs/1911.04474v3) | `ae7a0886fab35d7064d753bd6dab559fd80e2c42535ad3a0a167818c614a7937` |
| HotelRec | [2002.06854v1](https://arxiv.org/abs/2002.06854v1) | `5823cb334beddb319008f728fee04c5950b6ed0aa3d6aa2611ebd107be0efc68` |

三张查询图片本地位于 `data/raw/multimodal-eval/query-i01.png`、`query-i02.png`、`query-i03.png`，从对应原 PDF 用 `pdftoppm -f <页> -l <页> -png -r 100 -singlefile <PDF> <前缀>` 渲染的页面分别裁剪（左、上、右、下像素：`98,75,740,320`；`100,75,390,360`；`165,80,670,365`），以 Pillow RGB 和 Lanczos 分别缩放至 75%、125%、80%。三张都是架构图，可互为形态相近的干扰项；索引仍应包括全部发现的图表。查询图 SHA-256 分别为 `077138cd9dfeb7a460cba1d214a3b6610f04d63100d893d91a29956ad0682ad5`、`f0141b98ecbadd4aa430ec71eb6b5cf20e22e622ca9a1d0cc2a79d80480ba868`、`18e7edd4b7973474e3c6538ec318104ad65c5a3e4f1e57f9e3d4d54a913135cc`。裁剪查询仅供本机检索，不是可信证据图或页面坐标。

## 结果记录

创建空白逐例记录（输出必须为新文件）：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/multimodal_eval_record.py --output data/raw/multimodal-eval/results-template.json
```

已有预测可使用 `--predictions` 输入 JSON，其中 `cases` 是以案例 ID 为键的对象，各项包括 `retrieval_complete`、按顺序排列的 `top5`（每项含 `paper_id`、`page`、`figure_number` 等）、`answer`、`citations`、`citation_location`、`semantic_support`、`usage`、`elapsed_seconds`、`failure`。传入证据实体的 `figure_number` 必须包含 Figure/Table 和编号；脚本按论文、物理页码和完整图表编号判断视觉命中。如果编号无法可靠关联，则结果为 null；正文命中需要按对应段落位置独立核验，脚本不以同页命中冒充。`citation_location` 和 `semantic_support` 需独立逐例复核，默认 null；引用结构合法不等于答案受支持。不存在预测、usage 或证据时保持 null，不填零。混合题分别检查图和正文。

源页可使用 `data/raw/multimodal-eval/check-<paper>-p<page>.png` 查看，具体页、图号与问题在 `cases.json`。模型索引的 `paper_id + chunk_id` 应通过导入证据目录映射到同页同图表；重复、缺编号或多个匹配均需人工核对，不可仅凭图号猜测。评估时保留原始有序候选和逐案结果；如执行失败，记录错误、实际请求数和实际返回 usage，未执行为 unknown。

`scripts/run_multimodal_acceptance.py --help` 列出真实验收参数。它对每道问题固定添加 `Only paper <paper_id>:` 范围前缀，关闭远端遥测，使用同一个 reasoner 和最多 60 次的持久预算，并在发送前记录占位。使用前先完成本地索引、核对地址与固定题集；真正调用会消耗现有额度。固定的 `data/raw/multimodal-eval/acceptance-budget-v1.json` 只能独占创建一次，跨输出路径也不会重置；中断保留 unknown，不能删除后重跑来规避总预算。任何后续执行前须人工核算所有占位并遵守原 60 次上限。该运行器不自动判定答案语义支持，失败仍占预算。`u01` 的原 PDF 文字层 6 页扫描未见 SHA-256 字样或 64 位十六进制串，此扫描无法证明其他材料不存在或 PDF 图片中没有该信息。

固定文本回归：本轮在正常权限 `.venv-gpu` 中运行 `python -m pytest -q --basetemp=.cache/pytest-mm-baseline`，结果 276 passed、1 skipped、73 warnings，24.36 秒；受限环境 `_ssl` DLL 初始化失败。全量测试运行时其他代理可能已开始准备修改，不能声称该结果是严密冻结源码的历史性能基线；既有冻结文本检索指标以 `reports/FINAL_ACCEPTANCE.md` 为准。
