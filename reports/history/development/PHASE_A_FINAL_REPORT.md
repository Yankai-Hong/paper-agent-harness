# Phase A 最终回归报告

完成状态：代码、来源规则标签补齐、实际回归分别按下表核验；人工金标及独立 Judge 验证未完成，不能宣布全部验收完成。历史 acceptance 结果没有替代本轮结果。
基准 Git HEAD：4368a62a4ba15160fb5ff452194fd10bd2737c4e，已通过 git ls-remote 核对远端一致。保留原有 output/、tmp/。未提交修改用文件 SHA-256 绑定。
初次冻结后发现 Rewrite 错误分母、Memory 仅来源检查点两处缺陷并修复；初次 Intent/Rewrite 保留，最终表采用 final-intent/final-rewrite 重跑。其余运行开始时已加载修复后的模块，各 run_config.json 保存实际源码哈希。

| 模块 | 样本总数 | 执行失败 | 指标与计分分母 | 已知 token（含该目录 Judge） |
|---|---:|---:|---|---:|
| intent | 100 | 0 | accuracy: 86/100 (86.00%) | 54188 |
| rewrite | 90 | 0 | entity_retention_rate: 96.67%，分母 60 | 102916 |
| boundary | 55 | 2 | decision_accuracy: 48/55 (87.27%) | 1928413 |
| memory | 80 | 0 | checkpoint_accuracy: 73/80 (91.25%) | 2348375 |
| e2e | 120 | 0 | execution_success_rate: 120/120 (100.00%) | 2643006 |
| ablation/memory-bounded | 20 | 0 | checkpoint_accuracy: 20/20 (100.00%) | 496736 |
| ablation/memory-full | 20 | 0 | checkpoint_accuracy: 20/20 (100.00%) | 538761 |
| ablation/memory-recall | 20 | 0 | checkpoint_accuracy: 20/20 (100.00%) | 662876 |
| ablation/memory-semantic | 20 | 0 | checkpoint_accuracy: 20/20 (100.00%) | 631193 |
| ablation/reflect-disabled | 12 | 0 | execution_success_rate: 12/12 (100.00%) | 160312 |
| ablation/reflect-enabled | 12 | 0 | execution_success_rate: 12/12 (100.00%) | 183977 |
| ablation/reflect-no-second | 12 | 0 | execution_success_rate: 12/12 (100.00%) | 188298 |
| ablation/reflect-retrieve-again | 12 | 0 | execution_success_rate: 12/12 (100.00%) | 192707 |
| retrieval/bm25 | 100 | 0 | Recall@5: 66.26%，分母 100 | 0 |
| retrieval/dense | 100 | 0 | Recall@5: 73.94%，分母 100 | 0 |
| retrieval/hybrid | 100 | 0 | Recall@5: 74.24%，分母 100 | 0 |
| retrieval/rerank | 100 | 0 | Recall@5: 77.61%，分母 100 | 0 |

完整指标、95% CI、分母、未知数、P50/P95、实际返回模型与调用数见 phase-a/summary.json；逐例失败见 phase-a/failures.jsonl，配置差异/配对差值见 paired-comparisons.json。CI 跨零或仅少量会话不得声称增益。
Intent/Boundary 为标签一致率，非人工准确率；Memory 为结构检查点通过率，非事实语义正确率。E2E execution/task/pipeline 分开，未知任务质量不计成功，保留分母。引用结构合法不证明语义支持。
API 使用 .env 既有 OpenCode deepseek-v4.1-flash，显式 4096/disabled，不改凭据。真实 usage 计数来自响应，非字符/4 估算；币值单价与账户剩余额度不可获取，费用未知。
首次沙箱运行 SSL DLL 初始化失败，获准使用原环境后恢复。原 Intent 100 次请求预算耗尽留有失败，最终重跑预算 200，均有限重试。完整命令及运行时间见 full/jobs.json 与各 run_config.json。
消融沿用已有 v0.3 小子集（Reflect 12 turns、Memory 20 turns），与 v0.6 全量分开统计，不能混合总分；reflect-enabled 与 reflect-no-second 是同参数重复组。
未完成：真正人工金标/抽查 Judge 一致性；自然长会话的精确事实更新和归档精确率；无独立充分性标签的 Reflect 子集质量；未知服务端超时成本及金额。

复现：见 docs/evaluation.md 的 Phase A 命令。原资产直接复用；不下载模型或重建索引。

E2E 质量口径：
- 行为标签一致率：84/120；任务与综合 Pipeline 已满足条件下界均 30/120，任务质量未知 54。不能将上表 execution 100% 解释为任务质量 100%。
- 同模型辅助答案质量：80.83%，95% CI [0.7333333333333333, 0.875]；不是独立人工准确率。

质量与成本分开比较：
| 运行 | 辅助答案质量 | Pipeline token（不含 Judge） | P50/P95 阶段耗时 ms |
|---|---:|---:|---|
| memory | 82.50% | 1826662 | [18442.93000001926, 35164.14520002433] |
| e2e | 80.83% | 1549399 | [10977.009350142907, 22507.52218490815] |
| ablation/memory-bounded | 85.00% | 374583 | [35038.520099944435, 120357.4262248585] |
| ablation/memory-full | 95.00% | 416977 | [18522.098249930423, 26057.136040070338] |
| ablation/memory-recall | 90.00% | 543640 | [21178.87499992503, 33145.10199486395] |
| ablation/memory-semantic | 95.00% | 507110 | [24716.717900068033, 56121.64848505638] |
| ablation/reflect-disabled | 91.67% | 108438 | [12524.691500118934, 26864.419195119986] |
| ablation/reflect-enabled | 75.00% | 134847 | [11886.401699914131, 19384.716769953957] |
| ablation/reflect-no-second | 75.00% | 139474 | [11032.28460001992, 19857.221980125178] |
| ablation/reflect-retrieve-again | 75.00% | 139709 | [16261.911850073375, 35081.173829903106] |

延迟异常：本轮部分 Judge 请求记录约 8,895 秒 elapsed，远超配置的 120 秒；最终触发传输失败并有限重试。具体外部服务/宿主暂停原因未确认，不能据此推断正常吞吐或宣称硬实时超时保证；原始异常与未知成本均保留。
人工待复核仍阻塞最终科研验收。语义压缩与归档召回本次未证明质量增益；仅 2 个 Memory 会话簇的退化 bootstrap 区间即使不跨零，也不足以支持总体结论。
Reflect 的 12-turn retrieve-again 组本次二次检索触发为 0/12，因此不能评价二次检索后的证据改善率。Memory 三个压缩组均压缩 12/20 轮，召回组 10/20 轮使用归档；这是实际触发记录，不是召回精确率。见 full/ablations/trace-diagnostics.json。
Recall/MRR/nDCG/Hit@10 的置信区间、BM25/Dense 互补案例与 RRF/重排逐题变化见 full/retrieval/extended-statistics.json；70-turn 子集从完整120轮计分，见 full/e2e-70-view.json，不声称另跑70轮。
Claim 覆盖错误 d28-t3 的第9条声明保留 unknown，没有把 Judge 的表面 supported 输出算成功。全部8条证据不足和该失败见 claim-failures.jsonl；44条固定抽样/异常审阅队列待人工确认。零矛盾和100%结构有效的退化bootstrap区间不证明总体风险为零。

成本账本（含失败、重跑、Judge）：已保存 2493 次请求、已知 10243143 token，19 次 usage 未知；另 smoke 2 次/575 token。金额与剩余额度未知。见 phase-a/cost-ledger.json。

最终单元/可靠性测试：68 项，失败 0；见 phase-a/final-tests.xml。包含超时、429/503、非法 JSON、Schema 修复、引用错误、摘要降级、归档失败、幂等恢复及新增 Claim/分母/审核测试。

辅助质量与消融配对（不是人工准确率）：
```json
{
  "reflect-enabled minus reflect-disabled": {
    "common": 12,
    "missing_a": [],
    "missing_b": [],
    "status": "provisional_same_model_judge",
    "answer_correct_delta": {
      "mean": -0.16666666666666666,
      "n": 12,
      "clusters": 3,
      "bootstrap_95_percentile_interval": [
        -0.25,
        0.0
      ]
    }
  },
  "reflect-retrieve-again minus reflect-enabled": {
    "common": 12,
    "missing_a": [],
    "missing_b": [],
    "status": "provisional_same_model_judge",
    "answer_correct_delta": {
      "mean": 0.0,
      "n": 12,
      "clusters": 3,
      "bootstrap_95_percentile_interval": [
        -0.25,
        0.25
      ]
    }
  },
  "memory-bounded minus memory-full": {
    "common": 20,
    "missing_a": [],
    "missing_b": [],
    "status": "provisional_same_model_judge",
    "answer_correct_delta": {
      "mean": -0.1,
      "n": 20,
      "clusters": 2,
      "bootstrap_95_percentile_interval": [
        -0.1,
        -0.1
      ]
    }
  },
  "memory-semantic minus memory-full": {
    "common": 20,
    "missing_a": [],
    "missing_b": [],
    "status": "provisional_same_model_judge",
    "answer_correct_delta": {
      "mean": 0.0,
      "n": 20,
      "clusters": 2,
      "bootstrap_95_percentile_interval": [
        -0.1,
        0.1
      ]
    }
  },
  "memory-recall minus memory-semantic": {
    "common": 20,
    "missing_a": [],
    "missing_b": [],
    "status": "provisional_same_model_judge",
    "answer_correct_delta": {
      "mean": -0.05,
      "n": 20,
      "clusters": 2,
      "bootstrap_95_percentile_interval": [
        -0.1,
        0.0
      ]
    }
  }
}
```
