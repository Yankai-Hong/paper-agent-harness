# 评测数据集（data/eval/）

统一评测数据目录。生成脚本：`scripts/prepare_eval.py`（无模型调用）。数据版本、哈希与标签来源见 `manifest.json`。

## 标签来源与可信度

- **retrieval.jsonl**：直接复用 [QASPER v0.3](https://huggingface.co/datasets/allenai/qasper) 官方标注（CC BY 4.0），`gold_units`/`gold_alternatives` 为官方证据段落，**可作为检索指标的 ground truth**。
- **dialogs.jsonl / intent.jsonl / query_rewrite.jsonl / boundary.jsonl / memory.jsonl**：由 `data/dialogues-draft/` 派生，`review_status=draft_unreviewed`。其中的 `intent`、`decision`、`required_entities` 等是**模板派生的临时标签**，不是人工审核的 gold。任何基于这些标签的正确率/一致性结论都必须先经独立人工复核，否则只能作为“与草稿标签的一致率”报告，不能作为效果指标。
- **memory.jsonl** 的 `checkpoints` 目前为空模板，需在运行真实流水线后由人工填入期望的实体/事实检查点。

## 各文件 Schema

### intent.jsonl
```json
{"id": "d01-t1", "question": "...", "intent": "search|fact|explain|compare|out_of_scope",
 "review_status": "draft_unreviewed", "label_source": "..."}
```

### query_rewrite.jsonl
```json
{"id": "d01-t2", "question": "...", "history": ["..."], "required_entities": ["paper-id"],
 "expected_query": null, "review_status": "draft_unreviewed", "label_source": "..."}
```
`expected_query` 为空表示待人工填写期望的独立化查询；评测时 `eval rewrite` 会跳过缺失 `expected_query` 的语义保持分，只报告实体保留与指代消解诊断。

### retrieval.jsonl（QASPER）
```json
{"query_id": "...", "query": "...", "gold_units": ["paper::paragraph"], "gold_alternatives": [...],
 "source_paper_id": "...", "paper_ids": ["..."], "scope": "known_paper", "annotation_source": "QASPER official ..."}
```

### boundary.jsonl
```json
{"id": "d01-t4", "question": "...", "decision": "answer|refuse|clarify",
 "review_status": "draft_unreviewed", "label_source": "..."}
```

### dialogs.jsonl（30 组 × 4 轮）
```json
{"turn_id": "d01-t1", "session_id": "d01", "turn": 1, "question": "...",
 "review_status": "draft_unreviewed", "provisional_intent": "...", "provisional_decision": null,
 "source_query_ids": ["..."], "source_paper_ids": ["..."], "reference_units": ["..."],
 "annotation_note": "Template-derived follow-ups; source evidence is NOT gold for new questions."}
```

### memory.jsonl
```json
{"id": "mem-d01", "session_id": "d01",
 "checkpoints": {"entities": {"it": "paper-id"}, "facts": {"key": {"text": "...", "evidence_ids": ["c-..."]}}},
 "expected": {"resolved_paper_id": "paper-id"}, "review_status": "draft_unreviewed", "label_source": "..."}
```

## 评测协议

1. 固定语料快照（`data/qasper/papers.jsonl`，manifest 记录 archive SHA256）、分块配置（BGE tokenizer 500/80）、query 集、metadata filter、Top-K（召回 30、证据 5）、去重与证据 ID 规则。
2. 同一 query 集、同一模型配置、同一 seed。多轮对话以 session 为独立抽样单位，不能把 turns 当作完全独立样本。
3. 检索四组（BM25 / dense / hybrid / rerank）见 `eval retrieval`；意图与改写见 `eval intent`/`eval rewrite`；边界见 `eval boundary`；记忆见 `eval memory`；端到端加 Judge 见 `eval e2e --judge`。
4. LLM 生成的标签不得直接作为 ground truth。正式结论前必须人工抽查（计划要求）。
