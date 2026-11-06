# V4 过渡实现的旧题开发复验：独立 agent 原页核查

核查者是未参与该轮实现或模型调用的另一 agent，**不是人工复核，也不是盲审**。资料为旧的 BERT / Transformer-XL、ELMo / Sentence-BERT 题集和原 PDF，`runs/multimodal-v4-final-development-20261101/`、`runs/multimodal-v4-h07-development-20261101/` 内的实际预测及本地 `claim-audit`。本次未查看新保留集解析、排名或模型结果，也未修改代码、题集或调用模型。V4 目录名不代表本次数据是最终冻结源码的独立验收；它是执行时过渡实现的定向开发复验，不和旧题首次成绩或新保留集结果混算。

判断严格区分输出事实、实际引用的页/bbox 能否直接支持、模型的自检结论和最终决策。页码为 PDF 物理页；图框与位于框外的图注不能混同。原文核查只是本 agent 的判断，不等于独立人工金标。

## BERT / Transformer-XL 开发案例

| 案例 | 结果与逐声明、原页核查 |
|---|---|
| v202 | BERT 第 3 页 Figure 1 排 Top-5 第 1，正文 `v-ad65095d0a08dc5ddcd489e7` 排第 2；最终 `claim_self_check_failed` 拒答，无最终声明。原始生成六条中：`fig1_two_stages` 引同页图 `v-a95e7d5f2b33cf30e5636ebd` 与正文，明确的预训练/微调两阶段受支持；`fig1_pretrain_left` 引该图，未标注句对、NSP、Mask LM 可见；`caption_init_same_params` 和 `caption_all_ft` 引独立图注 `v-bcb662d63ac62b1b64676acc`，同模型初始化与微调全部参数可见；`text_separate_models` 引正文，各下游任务单独模型但相同初始化受支持。`fig1_finetune_right` 引图描述问答等内容，大部分可见，但声明中的 `MLN\|` 与图上 **MNLI** 不一致，该额外任务标识按字面无法由图支持，可能是生成拼写错误而不是原图内容。首次及再次同模型自检都把这一条判 unsupported，最终整体拒答；不能把本例的自检判定描述成已证实误判，但多余细节阻断了其余足以回答原题的内容。图注、正文和图像页/bbox 各自对应原页。 |
| v205 | 原题明确针对 Transformer-XL 的 Figure 2 和 3.2 节，但系统在 NLU 后直接要求明确论文，未检索、未作答；是任务级不必要澄清。原论文第 4 页 Figure 2 和第 3 页 3.2 节可支持旧状态缓存、扩展上下文及停止梯度；无最终引用，不可标作引用支持通过。当前本地 request/trace 未保留 NLU 原文的完整内容，具体由 NLU 自身或后续范围守卫触发不可单凭本审查断言。 |
| v207 | P19 第 6 页 Table 1 `v-aaa51bde12f46c1511122763` 排 Top-5 第 2；`standard_ppl` 的 24.0 PPL、151M 与 `large_ppl` 的 18.3 PPL、257M 均与所引表格行一致，图框、页、论文及文件哈希对应。但两条声明都说 `Table 1 (WikiText-103 results, PPL column)`；所引唯一表格**图框内**有 PPL、模型、数值，却没有 `WikiText-103` 或 `Table 1` 字样，这些位于框外的独立图注 `v-760a365514f641643633c9e6`，本次均未引用。图表编号元数据仍标为 `docling_link_unverified`。故核心数字正确，但按局部区域直接支持的严格口径，两条的数据集/表号条件仍需单独引用图注，不能称每项条件均由图框直接支持；该情况不是旧版本额外写入 test 的同一问题。 |
| v208 | BERT Figure 2 图在最终 Top-5 第 1、Transformer-XL Figure 2 图第 4；两篇目标图均命中。`bert_fig2_sum` 引 BERT 第 5 页图注 `v-2fbf5a8801a0edf308bc1a98`，三种嵌入求和受支持；`bert_fig2_visual` 引图 `v-aa353350eacefb1f8c95708d`，E0–E10、EA/EB 和示例 token 表示可见；`txl_fig2_caption` 引 P19 第 4 页图注 `v-43e329fad972aee9f09d857f`，图名和 segment length 4 受支持；`txl_fig2_visual` 引图 `v-2a91c8d9e30bafb52e70bf11`，两个阶段、旧段 `Fixed (No Grad)`、新段、绿色连线及 Extended Context 均可见。四条实际引用的页、框和来源与原页对应。答案通过图上固定旧段与新段相连、扩展上下文表达跨段机制，但没有明确写“缓存并复用前一段隐藏状态”；若题目要求该机制的完整文字解释，应记录为图示信息支持、显式阐述不完整，而不是自动当作完整的缓存机制描述。 |
| v209 | 无范围题中 BERT 第 4 页正文排 Top-5 第 1、第 5 页 Figure 2 排第 2。两条分别引用第 5 页图注 `v-2fbf5a8801a0edf308bc1a98` + 图 `v-aa353350eacefb1f8c95708d`，以及第 4 页正文 `v-7e0a75abc321c4237fa7d664`；BERT 和三种嵌入相加的识别均获这些局部区域支持。该次回答称 BERT 及论文 ID，没有附加完整标题的独立事实声明；证据上的 `source_metadata.title` 明确标 `unverified` 且 `scope=paper_identity_not_cited_region`，应与局部文本分开看，不把完整标题当作已在图注或图框逐字出现。唯一性仅限两篇论文语料。 |
| v210 | 新会话中两篇论文各有不同 Figure 1；未检索即要求明确论文符合题设，不是误澄清。无最终主张或引用；存在可靠历史上下文时需另判。 |

v207、v208、v209 的实际生成与同模型逐项 verdict 均已保存且判 supported；`unknown_visual` 数字检查不证明图中文字。以上局部核查直接比对原页，不能因 self-check 全通过而补出框外条件或缺失的机制。无最终回答的 v202、v205、v210 不赋予最终声明的语义通过率。

## Sentence-BERT h07 条件检查开发复验

`runs/multimodal-v4-h07-development-20261101/` 中最终答案是 `claim_condition_unknown` 拒答；原始三条声明与所引正文可从本地审计追溯：

1. `bert_time` 说在 10,000 个句子中找最相似对需要约 5,000 万次推理、约 65 小时；所引 D19 第 1 页摘要前段 `v-f7ae3b58ddd9328d21034749` 直接给出数量、推理规模和时间，摘要后段 `v-9a9b892764a8e2dc0b87af34` 支持比较。
2. `sbert_time` 说 SBERT 把 `this effort` 减至约 5 秒，并说明 10,000 条嵌入和约 0.01 秒余弦计算；它自己的引用包括 D19 第 2 页 `v-b190d57e172001718f5b6a77`，该段明确包含 **10,000 句集合、寻找最相似对、65 小时转为 10,000 个嵌入约 5 秒及余弦约 0.01 秒**，足以在该引用段内解析对应条件；另引第 1 页摘要后段。
3. `comparison` 的摘要 65 小时到约 5 秒、保持 BERT 准确率，由其所引第 1 页摘要后段支持。

三条原始声明各自的数字和实际任务语境可由所引段落核查；但条件检查把第二条因 `this effort` 标为 `context_dependent`，并在模型逐声明自检前整体拒答（`verdicts=null`）。这是本题可从**自身已引用证据**消解指代而仍拒答的保守规则误拒，不能与无支持声明被正确拒绝混同，也不能据此推断所有代词均可放行。两个实际原页段落的 PDF 页和 bbox 与本地引用信息一致。该运行保留了原始声明及拒答原因；它是旧题诊断，不能视为新保留集成绩。

所有原页核查和裁图判断仍需独立人工复核；本报告未覆盖未列案例或其他图表，未解决的关联和语义保留 unknown。
