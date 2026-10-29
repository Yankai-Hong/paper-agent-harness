# 追踪和范围修复后的模型对照：执行前协议

固定数据：QASPER v0.6 原顺序前两个不同论文；一条单论文问题、一条组合两篇问题、一个作者构造的训练数据 SHA-256 缺失证据探针。每条分别运行拆分关闭与开启，合计 3 对、6 轮；使用本地既有 BM25 索引。数据与脚本将在结果中记录哈希。缺失探针先检查目标论文的 chunk 中是否出现 64 位十六进制字符串；这个规则不能证明语义完整性。

严格上限：18 个实际模型请求，每轮最多 3 次；单次输入不超过 30,000 字符、输出不超过 4096 token、45 秒超时，每轮总限 140 秒；理论输出上限 73,728 token，输入是字符上限而非 tokenizer 数。thinking=disabled，传输重试、Schema 修复、Judge 及额外补检索均为 0。费用取决于实际 provider 计价，无法由本地预算确定。仅发送公开 QASPER 问题与检索到的公开论文片段，不读取个人记忆；遥测明确关闭，不修改 .env。

所有错误和未完成组保留；从实际 provider usage 汇总 token，缺失为 unknown。论文目标覆盖与官方证据段落命中分开，均不是语义支持；答案完整性、无依据结论和误拒答没有可靠独立标签时保留 unknown。试验不因结果不理想扩大规模。

执行顺序：先完成代码修复及测试，再运行 `scripts/revision_model_comparison.py --output reports/revision/model-comparison-followup`。目录必须不存在；脚本落盘预算与各轮结果，不覆盖历史。先前 model-pair/summary.json 的 `budget.actual_requests=6` 是上限而非实际；逐组 2+3=5，原文件保持不变。
