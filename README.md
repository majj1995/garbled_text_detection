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

仓库内置 `data/generated/smoke-a`、`data/generated/smoke-b` 与
`artifacts/train-smoke`，用于克隆后的 CPU 冒烟验证；`data/raw` 中同时附带来源已锁定、
许可已登记的字体和常用汉字表。这批模型与数据只证明工程链路可运行，不代表生产效果。
模型权重通过 Git LFS 保存，克隆机器需先安装 Git LFS，并在克隆后执行：

```bash
git lfs install
git lfs pull
```

目录级忽略规则仍然保留，因此后续新增的业务原图、完整训练集和实验产物不会被自动加入
Git；如需发布新的已审计版本，应显式选择文件并复核许可、隐私和体积。

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

## 训练后诊断（不重训）

如果低误报下召回不足，使用同一个全局阈值分析五类合成异常、导出错例，并重放训练
采样以检查同字正样本配对覆盖率。不需要启动 PaddleOCR，不会重新下载预训练权重。

在项目根目录执行；下面使用物理 2 号卡，如使用 3 号卡只改 `CUDA_VISIBLE_DEVICES`：

```bash
CUDA_VISIBLE_DEVICES=2 uv run poor-word evaluate diagnose-glyph \
  --manifest data/generated/mvp-holdout/manifest.parquet \
  --train-manifest data/generated/mvp-v1/manifest.parquet \
  --artifacts artifacts/glyph-mvp-v1 \
  --device cuda:0 \
  --output-dir artifacts/diagnostics-mvp-v1
```

默认在当前诊断集上选择 `FPR <= 0.0001`（即 0.01%）时召回最高的全局阈值，
不会按异常类型分别调阈值。可用 `--threshold <完整精度阈值>` 固定阈值；固定阈值即使
超出 FPR 上限也会原样报告，不会静默换阈值。此处是开发诊断，不能替代独立验收。

输出目录必须是新目录，重跑请使用新名字，避免覆盖已有结果。训练清单会与训练指标和
原型库来源 SHA-256 对齐，模型、原型库及字符表也会核验。读取图片前会验证路径不越出
数据目录。推理和原型评分期间会持续打印进度。

主要输出：

- `diagnostics.md`：总体结果、五类异常召回表、训练采样配对覆盖率。
- `diagnostics.json`：完整诊断配置、指标、来源文件哈希和逐轮配对统计。
- `scores.parquet`：逐样本原型距离、最近原型字、预测和 TP/FP/TN/FN，保留原始清单字段。
- `examples/index.md`：可点击的错例原图和掩码索引；原图按字节复制，不做修图。
- `examples/index.json`：错例来源字、算子、分数、原始路径和图像/掩码 SHA-256。
- `examples/FN/<operator>/`：默认每类最多 10 张随机漏检样本，固定种子可复现。
- `examples/FP/`：默认最多 50 张误报，超过上限时按异常分数从高到低导出。

可用 `--examples-per-kind`、`--max-false-positives`、`--seed` 调整抽样。
FN 掩码白色表示被合成器修改的像素；FP 掩码白色表示正常字前景，不是异常定位。
来源字只是合成起点，不能当作修改后图像的人工金标。请人工检查漏检图是否仍是合法字、
是否变成另一个合法字、变换是否过轻，以及是否符合业务异常定义；抽查不能自动估算全量
标签错误率。

采样重放使用 checkpoint 中的 seed、batch size、epochs/max steps 及训练清单，
只读取元数据、不读训练图片、不训练模型。`eligible_normal_fraction` 表示正常样本抽取
次数中，同批存在至少一个同字正常样本的占比；`batches_without_positive_pairs_fraction`
表示没有任何同字正常配对的批次占比。它不是历史损失日志，也不能独立证明模型学习或
坍塌。报告对比实际记录步数，跨代码/PyTorch 版本需谨慎解释重放结果。

先查看 `diagnostics.md` 的“按异常类型统计”和“训练采样配对覆盖率”，再打开
`examples/index.md` 抽查；无需把业务图片或模型传出内网。

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
`source_image_sha256`、非空可信 `duplicate_group_id`、真实/折叠清单字段及其 SHA-256、
两个独立模型来源的风险分数与检查点
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
字符级硬去重只使用相同 `crop_sha256` 或显式 `duplicate_group_id`；同一原图中的不同字符
不会因为共享 `source_image_sha256` 被删除。分歧来源的 model ID 和 checkpoint SHA-256
都必须不同。默认要求合规开发数据完整覆盖连续五折 `0..4`，可用 `--fold-count` 显式调整。

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
  --review-queue data/real/review-queues/queue-<queue-version> \
  --review-labels review-complete.jsonl \
  --reviewed-gold data/real/reviewed-gold/gold-<version>/gold-crops.parquet \
  --import-audit data/real/reviewed-gold/gold-<version>/import-audit.json \
  --base-real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --output-dir data/real/versioned/seed-v2
```

`mining-yield` 同时绑定 Task 3 的完整 review queue、原始回传 labels、import audit 和可信
gold，重新验证队列内容版本、labels SHA-256、每一条 PASS/BLOCK/REVIEW 的成员关系及
模型/图片/裁剪链路，然后只写 `dataset-version.json` 和 yield 审计；它不创建、
复制或改写 gold。分母明确为该次 import audit 的 `accepted_label_count + review_count`。
候选是有偏采样，因此该 yield **不是现网异常率估计**；`0.001` 只作为现网基准率上下文，
本命令不批准生产阈值。该流程是 CPU 元数据处理，无 L20/GPU 特殊要求。

## 真实字符 OOF、图像 MIL 与阶段报告

以下命令在 NVIDIA L20 的主 Python 3.12 uv 环境执行；PaddleOCR v5 server 仍在上文的
隔离环境通过回环 HTTP endpoint 提供 OCR。真实字符模型按五折训练，每个开发字符只由
未见过其所属折的模型评分：

```bash
uv run poor-word train real-oof \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet \
  --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet \
  --synthetic-manifest data/generated/mvp-v1/manifest.parquet \
  --adapted-checkpoint artifacts/glyph-real-adapt-v1/encoder.pt \
  --output-dir artifacts/real-oof-v1 \
  --device cuda --epochs 10 --batch-size 64

uv run poor-word train real-nested-oof \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet \
  --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet \
  --synthetic-manifest data/generated/mvp-v1/manifest.parquet \
  --adapted-checkpoint artifacts/glyph-real-adapt-v1/encoder.pt \
  --output-dir artifacts/real-nested-oof-v1 \
  --device cuda --epochs 10 --batch-size 64

uv run poor-word train mil-oof \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet \
  --nested-feature-dir artifacts/real-nested-oof-v1 \
  --output-dir artifacts/mil-oof-v1 \
  --device cuda --epochs 30 --batch-size 32
```

`real-nested-oof` 为每个 outer MIL fold 生成一份字符特征。对 outer fold `k`，验证 fold
`k` 的字符分数来自排除 `k` 的模型；训练中 fold `j != k` 的字符分数来自同时排除 `k`、`j`
的模型。该命令原子发布带有每个模型 checkpoint/metrics/scores 哈希的
`nested-manifest.json`。`mil-oof` 拒绝普通单层字符 OOF，仅消费这个嵌套清单。

`mil-oof` 在单一 staging root 中依次运行五折。每折 `image-scores.parquet` 对该折每张
合规验证图片恰有一行，包括没有检测到字符的 zero-character bag；命令严格验证后才原子
发布 `image-oof.parquet`、`model-inventory.json` 和 `metrics.json`。任一折失败会清理整个
staging root。`attention-candidates.parquet` 仍只是人工诊断候选，不能冒充全量 OOF 分数。

PP-OCRv5 server 的字符基线清单必须对每个字符 OOF ID 恰有一行，包含 `crop_id`、
`image_id`、`fold`、`decision`、`anomaly_kind`、单字符 `text`、`ocr_confidence`、
`ocr_model_id=PP-OCRv5_server_rec`、对应 `ocr_audit_sha256`，以及可选的独立
`visual_anomaly_score`。合法 CJK 字即使不在 3500 常用字表也只标记
`out_of_catalog/needs_review`；没有独立视觉异常证据时不得自动 BLOCK。

```bash
uv run poor-word ocr audit \
  --endpoint http://127.0.0.1:8765 \
  --image-dir data/real/seed/images --warmup 10 --runs 30 \
  --output artifacts/ocr-audit-l20.json

uv run poor-word evaluate real-seed \
  --real-manifest data/real/versioned/seed-v1/manifest.parquet \
  --fold-manifest data/real/versioned/seed-v1/split-v1/folds.parquet \
  --crop-manifest data/real/versioned/seed-v1/crops-v1/crops.parquet \
  --gold-manifest data/real/versioned/seed-v1/gold-v1/gold-crops.parquet \
  --character-oof artifacts/real-oof-v1/oof/oof.parquet \
  --image-oof artifacts/mil-oof-v1/image-oof.parquet \
  --ocr-manifest artifacts/ocr-character-scores.parquet \
  --ocr-audit artifacts/ocr-audit-l20.json \
  --common-chars data/raw/common_chars_3500.txt \
  --source-lock data/locks/common_chars_3500.lock.json \
  --dependency-lock uv.lock \
  --character-inventory artifacts/real-oof-v1/model-inventory.json \
  --image-inventory artifacts/mil-oof-v1/model-inventory.json \
  --nested-manifest artifacts/real-nested-oof-v1/nested-manifest.json \
  --prevalence 0.001 \
  --output-dir artifacts/real-seed-report-v1
```

输出为原子发布的 `report.json`、`report.md` 和记录两者 SHA-256 的
`report-provenance.json`。字符和图像均报告原始样本数、AUCPR、固定阈值混淆矩阵、Recall/
FPR 的 Wilson 95% 区间、异常类型召回及按 0.1% 基率重算的 PPV；`REVIEW` 只进入 review
rate，不进入监督指标。当正常负样本少于 10,000、真实 BLOCK 不足、缺少 normal replay，
或尚未通过受控流程运行 locked test 时，报告固定包含醒目的 `Not a pilot approval`，并把
结论写为 `inconclusive`，不会外推 FPR 或把挖掘/平衡样本 precision 当作生产 PPV。
