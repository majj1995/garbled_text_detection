# poor-word

面向电商商品图、营销海报和广告素材的 AIGC 中文乱码检测工程。

## 本地环境

```bash
uv python install 3.12
uv sync --all-groups
uv run poor-word doctor
```

本地 CPU 开发使用 Python 3.12。GPU 训练和 PP-OCRv5 server 审计在 NVIDIA L20
实验机的 Python 3.11 环境执行；CPU 单元测试不初始化 PaddlePaddle 或 CUDA。

若 Intel macOS 将 `.venv` 目录树标记为 hidden，Python 3.12 会跳过 editable
安装使用的 `.pth` 文件并报 `ModuleNotFoundError: poor_word`。可在同步后执行一次：

```bash
chflags -R nohidden .venv
```
工程兼容 Python 3.11–3.12。Intel macOS 本地环境使用最后一组提供对应 wheel 的
PyTorch 2.2.2；Linux/L20 训练环境使用 PyTorch 2.7.1，由根目录 `uv.lock` 管理。

PaddlePaddle 与 PyTorch 对 CUDA/NCCL 运行时存在互斥的精确版本约束，因此
PP-OCRv5 使用 `environments/ocr` 下兼容 Python 3.11–3.12、在 L20 固定 3.11 的
独立 uv 环境和锁文件。两个
运行时通过本地 HTTP/JSON 接口通信，不在同一 Python 进程中加载 GPU 框架。

```bash
# NVIDIA L20 / CUDA 12.6 实验机
uv python pin 3.11
uv sync --all-groups

# 独立 PP-OCRv5 server 环境
uv sync --project environments/ocr
```

在 L20 上启动只监听回环地址的 OCR 服务：

```bash
POOR_WORD_OCR_DEVICE=gpu:0 \
  uv run --project environments/ocr python environments/ocr/service.py
```

另一个终端从主环境执行能力与延迟审计：

```bash
uv run poor-word ocr audit \
  --endpoint http://127.0.0.1:8765 \
  --image-dir data/generated/smoke-a/images \
  --warmup 10 \
  --runs 30 \
  --output artifacts/ocr-audit-l20.json
```

PaddleOCR 标准 OCR 结果只有行级 `rec_texts/rec_scores/rec_polys/rec_boxes`。若服务
尚未提供真实字符框或解码前 logits，审计仍会写出 JSON，但以退出码 2 标记能力缺口；
系统不会把等宽切分框伪装成模型输出。

## 锁定数据源

来源声明记录在 `data/sources.toml`。`lock` 首次下载并记录重定向后的 URL、文件大小、
SHA-256、SPDX 许可证标识和生产使用决策；`fetch` 只接受与锁文件一致的内容。

```bash
uv run poor-word data lock --source-id common_chars_3500
uv run poor-word data lock --source-id noto_sans_sc_regular
uv run poor-word data fetch --source-id common_chars_3500
uv run poor-word data fetch --source-id noto_sans_sc_regular
```

下载内容保存在被 Git 忽略的 `data/raw`，可审计锁文件保存在 `data/locks`。

## 生成合成字形

```bash
# 10 个字符、全部五类异常算子的快速复现检查
uv run poor-word glyphs generate \
  --profile smoke \
  --seed 20260804 \
  --output-dir data/generated/smoke-a

# 3500 字 MVP 训练集（建议在算力机执行）
uv run poor-word glyphs generate \
  --profile mvp \
  --seed 20260804 \
  --output-dir data/generated/mvp-v1

# 独立合成 holdout；不同种子且不参与训练
uv run poor-word glyphs generate \
  --profile mvp \
  --seed 20260805 \
  --output-dir data/generated/mvp-holdout
```

每次运行输出 `manifest.parquet` 和 `run.json`。不满足变化像素与拓扑后置条件的
异常候选会计入 `skipped_count`，不会作为 BLOCK 样本写入。

## 训练字形基线

```bash
# 本机只做两步工程冒烟，ConvNeXt 特征层会冻结
uv run poor-word train glyph \
  --manifest data/generated/smoke-a/manifest.parquet \
  --epochs 1 --max-steps 2 --batch-size 4 --device cpu \
  --output-dir artifacts/train-smoke

# L20 完整训练
uv run poor-word train glyph \
  --manifest data/generated/mvp-v1/manifest.parquet \
  --epochs 20 --batch-size 256 --device cuda --pretrained \
  --output-dir artifacts/glyph-mvp-v1
```

训练输出 `encoder.pt`、`prototypes.npz/.json` 和 `metrics.json`；metrics 记录三项
损失、验证集最近原型准确率、合成 OOD AUCPR、Git/uv/manifest 哈希及运行时间。

## 低基率评估与报告

```bash
# 本地 CPU 工程冒烟；不提供 L20 审计时会明确记录 capability gap
uv run poor-word evaluate glyph \
  --manifest data/generated/smoke-a/manifest.parquet \
  --artifacts artifacts/train-smoke \
  --prevalence 0.001 \
  --device cpu \
  --output-dir artifacts/report-smoke

# L20 完整合成 holdout 评估，并纳入 OCR 能力审计
uv run poor-word evaluate glyph \
  --manifest data/generated/mvp-holdout/manifest.parquet \
  --artifacts artifacts/glyph-mvp-v1 \
  --prevalence 0.001 \
  --device cuda \
  --ocr-audit artifacts/ocr-audit-l20.json \
  --output-dir artifacts/report-mvp-v1
```

输出为 `report.json` 和 `report.md`。阈值表同时报告原始 TP/FP/TN/FN、召回、FPR
以及按现网异常率 0.1% 重算的 precision；不会使用合成集或平衡采样集的类别比例替代
现网基率。离线 MVP 门槛是 `FPR <= 0.01% 且 Recall >= 40%`，试运行门槛是
`FPR <= 0.0025% 且 Recall >= 50%`。当样本量不足以分辨相应 FPR 时，报告会写入
已知缺口；合成数据报告固定标注 `Not a production claim`。

本开发机没有 NVIDIA 运行时，只验证 CPU 冒烟链路。上文 PP-OCRv5 审计、完整训练
和 CUDA 评估命令必须在 NVIDIA L20 实验机执行。推荐的完整顺序是：锁定并拉取数据源、
生成独立训练/holdout 合成集、运行 CPU 测试、在 L20 审计 OCR、训练字形模型，最后生成
纳入 OCR 审计 JSON 的评估报告。

## CPU 质量门禁

```bash
uv lock --check
uv run pytest -m "not gpu" --cov=poor_word --cov-report=term-missing
uv run ruff check .
uv run mypy src
```

## 导入真实种子数据

真实数据使用 JSONL，一行一张图片；图片路径相对 JSONL 所在目录。图片级标签只表示
`NORMAL`（全部可见文字正常）或 `ABNORMAL`（至少一个区域异常），不能自动推导异常字符。
字符金标只接受带人工 `annotator_id` 的 `PASS`/`BLOCK`；未确认项写为 `REVIEW`。

```json
{"image_id":"case-001","image_path":"images/case-001.png","expected_sha256":"<64位小写SHA-256>","image_label":"ABNORMAL","split_role":"DEV","source_id":"business_seed","source_group_id":"upload-batch-1","license_id":"LicenseRef-Proprietary","production_allowed":true,"product_id":"product-1","campaign_id":"campaign-1","template_id":"template-1","label_provenance":"human-image-review-v1","training_eligible":true,"characters":[{"annotation_id":"case-001-char-1","box":{"x0":12,"y0":20,"x1":58,"y1":76},"decision":"BLOCK","annotator_id":"reviewer-1","anomaly_kind":"missing_stroke"}]}
```

```bash
uv run poor-word real-data import \
  --input data/real/seed/records.jsonl \
  --output-dir data/real/versioned/seed-v1
```

导入会解码每张图片、校验文件 SHA-256 与字符框边界，并验证来源许可和训练资格。
输出 `manifest.parquet`、`dataset.json`、`validation.json`；后者审计 60 张开发异常、
20 张锁定异常和 20 张困难正常的目标差额。`LOCKED_TEST` 或未获生产许可的数据不能
标记为可训练。已有输出内容不一致时导入器拒绝覆盖，以保护数据版本不可变性。

导入后按商品、活动、模板、上传/生成来源组、精确 SHA 重复和 pHash 近重复的传递闭包
分配五折。`source_id` 只表示许可来源，`source_group_id` 表示需要防泄漏的上传批次、
生成器或业务流。`--image-root` 是原始 JSONL 所在、包含 `images/` 的目录。

```bash
uv run poor-word real-data split \
  --manifest data/real/versioned/seed-v1/manifest.parquet \
  --image-root data/real/seed \
  --folds 5 \
  --seed 20260804 \
  --output-dir data/real/versioned/seed-v1/split-v1
```

输出 `folds.parquet` 和 `split-audit.json`。锁定测试统一为 `fold=-1`；若它与开发集
存在任何分组或近重复连通关系，命令直接失败，要求先修正数据划分。

## 字符裁剪与人工快审

字符裁剪只使用导入清单中的人工审核 `characters` 框，或同时提供 OCR JSONL 结果和明确
声明 `character_boxes_available=true` 的 OCR 审计。OCR JSONL 每行使用
`{"image_id":"...","result":<OcrResult>}`；其中 `result` 遵循项目的 OCR-neutral
schema。行级 OCR 框不会被等宽拆分或伪造为字符框。输出裁剪 PNG、`crops.parquet` 和
`crops-audit.json` 均为不可变工件，且包含原图 SHA-256 和裁剪坐标链路。

```bash
# 仅用已有的人审字符框
uv run poor-word real-data crops \
  --manifest data/real/versioned/seed-v1/manifest.parquet \
  --image-root data/real/seed \
  --output-dir data/real/versioned/seed-v1/crops-v1 \
  --padding 2

# 如使用 OCR，结果和能力审计必须成对提供
uv run poor-word real-data crops \
  --manifest data/real/versioned/seed-v1/manifest.parquet \
  --image-root data/real/seed \
  --output-dir data/real/versioned/seed-v1/crops-ocr-v1 \
  --ocr-results artifacts/ocr-character-results.jsonl \
  --ocr-audit artifacts/ocr-audit-l20.json
```

## 无标注真实裁剪自监督适配

适配必须同时给出不可变的 `crops.parquet` 和其来源的真实图片 `manifest.parquet`。
命令按 `image_id` 联接两者，在任何图片读取前拒绝重复或缺失 ID，并且只读取
`training_eligible=true` 且不属于 `LOCKED_TEST` 的裁剪。每次运行确定性地从这些合格裁剪中
留出一部分作漂移诊断；锁定测试裁剪既不训练，也不会被打开或用于诊断。

CPU 冒烟（不下载预训练权重）：

```bash
uv run poor-word train adapt-real \
  --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --prior-checkpoint artifacts/glyph-mvp-v1/encoder.pt \
  --output-dir artifacts/glyph-real-adapt-smoke \
  --device cpu --epochs 2 --batch-size 2 --max-steps 2
```

L20 完整适配：

```bash
uv run poor-word train adapt-real \
  --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --prior-checkpoint artifacts/glyph-mvp-v1/encoder.pt \
  --output-dir artifacts/glyph-real-adapt-v1 \
  --device cuda --epochs 20 --batch-size 64
```

输出的 `encoder.pt` 保留父检查点、两份来源清单的 SHA-256；`metrics.json` 记录种子、Git/
`uv.lock` 哈希、可用/训练/诊断裁剪计数、损失历史、耗时、坍塌防护与适配前后 held-out
embedding 漂移。两视图只使用亮度、对比度、灰度/颜色、噪声与模糊等风格变换，不作裁剪、
旋转、仿射或随机缩放。**适配后的检查点未针对 `BLOCK` 决策校准，不能直接作为 BLOCK
阈值或生产决策依据。**

快审队列的分数输入和模型分歧输入是 JSONL。分数行应含 `crop_id`、`image_id`、
`crop_path`、`risk_score`（0–1）、`style_id`、`score_model_id` 和
`score_artifact_sha256`；分歧行应含 `crop_id`、非负 `disagreement`、
`disagreement_model_id` 和 `disagreement_artifact_sha256`。队列按模型分歧、接近阈值的
候选和风格覆盖度确定性排序，并保留 `priority_rank`，导出版本化的 CSV、JSONL 与
contact sheet。人工回传 CSV 或 JSONL 时，每行必须带导出的精确
`queue_version`、`crop_id`、`label`（仅 `PASS`、`BLOCK`、`REVIEW`）和
`annotator_id`。

```bash
uv run poor-word review export \
  --crops data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --crop-root data/real/versioned/seed-v1/crops-v1 \
  --scores artifacts/real-crop-scores.jsonl \
  --disagreements artifacts/real-crop-disagreements.jsonl \
  --limit 500 --seed 20260804 \
  --output-dir data/real/review-queues

uv run poor-word review import \
  --labels review-complete.jsonl \
  --queue data/real/review-queues/queue-<queue-version> \
  --output-dir data/real/reviewed-gold
```

`REVIEW` 保持未解决状态，绝不会写入 `gold-crops.parquet`。已接受的 `PASS`/`BLOCK`
标签不可被其他审核者覆盖；增量导入时通过 `--existing-gold` 指向之前版本的
`gold-crops.parquet`。所有队列和 gold 输出目录以内容哈希版本化。

## 困难样本挖掘与下一数据版本

挖掘输入可以是不可变 Parquet 或 JSONL，每个字符候选都必须携带
`crop_id`、`image_id`、安全相对 `crop_path`、`crop_sha256`、
`source_image_sha256`、真实/折叠清单字段及其 SHA-256、两个模型的风险分数与检查点
SHA-256、MIL attention，以及 `style_cluster_id`、`style_novelty_score` 和
`style_is_unseen`。命令会先与冻结的真实和折叠清单联接；未知图片、来源/折叠不一致、
不合规来源和哈希错误都会失败。`LOCKED_TEST`/`fold=-1` 只计入跳过审计，绝不进入队列。

```bash
uv run poor-word real-data mine \
  --scores artifacts/mining-input-v1.parquet \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet \
  --overall-limit 500 --per-product-cap 25 --per-template-cap 10 \
  --per-source-cap 50 --seed 20260804 \
  --output-dir data/real/mining/candidates-v1
```

输出目录一次性发布 `queue.parquet`、`mining-audit.json`，以及可直接交给现有快审导出
命令的 `review-scores.jsonl` 和 `review-disagreements.jsonl`。队列覆盖正常图高风险误报、
双模型分歧、阈值带、异常 bag 高 attention 和新风格簇；重复内容及单一 product、template、
source 的过量候选会被抑制并记录。候选只有 `label_source=mining_candidate` 和
`is_gold=false`，不会包含人工决策或审核者字段，也绝不会直接写入 gold。

安全人工闭环如下：

```bash
uv run poor-word review export \
  --crops data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --crop-root data/real/versioned/seed-v1/crops-v1 \
  --scores data/real/mining/candidates-v1/review-scores.jsonl \
  --disagreements data/real/mining/candidates-v1/review-disagreements.jsonl \
  --output-dir data/real/review-queues --limit 500 --seed 20260804

uv run poor-word review import \
  --labels review-complete.jsonl \
  --queue data/real/review-queues/queue-<queue-version> \
  --output-dir data/real/reviewed-gold

uv run poor-word real-data mining-yield \
  --candidate-queue data/real/mining/candidates-v1/queue.parquet \
  --reviewed-gold data/real/reviewed-gold/gold-<version>/gold-crops.parquet \
  --import-audit data/real/reviewed-gold/gold-<version>/import-audit.json \
  --base-real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --output-dir data/real/versioned/seed-v2
```

`mining-yield` 只接受 Task 3 人工导入产出的可信 gold 和审计，校验新增字符确实来自候选
队列及其模型/图片/裁剪链路，然后只写 `dataset-version.json` 和 yield 审计；它不创建、
复制或改写 gold。分母明确为该次 import audit 的 `accepted_label_count + review_count`。
候选是有偏采样，因此该 yield **不是现网异常率估计**；`0.001` 只作为现网基准率上下文，
本命令不批准生产阈值。该流程是 CPU 元数据处理，无 L20/GPU 特殊要求。
