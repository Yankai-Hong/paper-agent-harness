# 冻结图片查询：LoRA

`fixture.json` 在解析新论文前固定，SHA-256 `e5b4386ed2a00b6826f9845bd9b7fd890e14c51db07ef2f7a99b1cd31d13bae3`。新论文此前未进入本项目开发索引；本助手在冻结前看过原页并制作题目，负例源自已见开发论文，所以本集不是人工金标或盲审。`results.json` 在实际评测后创建，不能当作检索输入。原论文、查询图、索引和运行数据不提交；原论文须自行从 [arXiv v2](https://arxiv.org/pdf/2106.09685v2) 取得并校验 fixture 内 SHA-256。负例还要求准备 `data/eval/multimodal-holdout-v3` 所述 ULMFiT 查询图及其哈希。

以下在项目根目录执行；所有输出都须为全新路径。使用本机已安装的独立 Docling/CLIP 环境及固定快照，必要时将设备改为 CPU。裁图生成会同时验证 fixture、PDF 和输出哈希：

```powershell
$env:TEMP = (Resolve-Path .cache).Path
$env:TMP = $env:TEMP
$py = '.\.venv-gpu\Scripts\python.exe'
$vpy = '.\.venv-mm-20261030\Scripts\python.exe'
$fixture = 'data/eval/multimodal-image-holdout-20261106/fixture.json'
$hash = 'e5b4386ed2a00b6826f9845bd9b7fd890e14c51db07ef2f7a99b1cd31d13bae3'
$raw = 'data/raw/multimodal-image-holdout-20261106'
& $vpy scripts/prepare_frozen_image_holdout.py --fixture $fixture --fixture-sha256 $hash --pdf "$raw/2106.09685v2.pdf" --output "$raw/lora-figure1-query.png"
& $vpy scripts/docling_ingest.py "$raw/2106.09685v2.pdf" --paper-id 2106.09685 --title 'LoRA: Low-Rank Adaptation of Large Language Models' --image-root "$raw/crops" --output "$raw/evidence.jsonl" --device cuda:0 --artifacts-path models/docling-mm-20261030-v2
& $py -m paper_agent.cli visual-collect "$raw/evidence.jsonl" --evidence-output "$raw/combined.jsonl" --text-output "$raw/text.jsonl"
& $py -m paper_agent.cli index "$raw/text.jsonl" --output indexes/lora-frozen-bm25-new
$v = @('--visual-index','indexes/lora-frozen-clip-new','--visual-root',"$raw/crops",'--visual-python',$vpy,'--visual-image-model','models/multimodal-v1/clip-image-v1','--visual-text-model','models/multimodal-v1/clip-text-v1','--visual-device','cuda:0')
& $py -m paper_agent.cli visual-index "$raw/combined.jsonl" @v
& $py -m paper_agent.cli clip-text-index "$raw/text.jsonl" @v --output indexes/lora-frozen-clip-text-new
& $py scripts/eval_frozen_image_holdout.py --fixture $fixture --fixture-sha256 $hash --index indexes/lora-frozen-bm25-new --clip-text-index indexes/lora-frozen-clip-text-new --visual-index indexes/lora-frozen-clip-new --visual-root "$raw/crops" --visual-python $vpy --image-model models/multimodal-v1/clip-image-v1 --text-model models/multimodal-v1/clip-text-v1 --device cuda:0 --output output/lora-frozen-new.json
```

真实模型问答会将所选论文图和用户查询图发送到 `.env` 配置的服务并产生费用；不要覆盖旧账本：

```powershell
& $py scripts/smoke_image_question.py --index indexes/lora-frozen-bm25-new --clip-text-index indexes/lora-frozen-clip-text-new --visual-index indexes/lora-frozen-clip-new --visual-root "$raw/crops" --visual-python $vpy --image-model models/multimodal-v1/clip-image-v1 --text-model models/multimodal-v1/clip-text-v1 --device cuda:0 --query-image "$raw/lora-figure1-query.png" --negative-image data/raw/multimodal-holdout-v3-20261101/v3img01.png --question 'What does the image show about which weights are frozen and which matrices are trainable?' --output runs/lora-frozen-new
```

首次评测的目标覆盖、失败、引用核查、用量和结果边界见 [修复报告](../../../reports/revision/MULTIMODAL-IMAGE-RETRIEVAL-REPAIR-20261106.md)。
