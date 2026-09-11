# V2：生成 → 训练 → 评估

本页使用已完成视觉校准的**原生笔画规则**。旧 `glyphs generate` 仍是 V1；
旧 `preview-v2` 是早期几何预览，二者都不是这里的 V2 训练入口。
无需再验收预览图片，也无需启动或重装 PaddleOCR。

## 0. 更新服务器代码和主环境

在已有、成功训练过的项目目录执行：

```bash
git pull --ff-only origin master
```

本次依赖未变化，不执行 `uv sync`。这里沿用主环境现有 Python，不修改 `.python-version`，
也不操作 `environments/ocr`。下方显式选用 `.venv/bin/python` 并禁止重复同步，避免运行时切换
已工作的解释器。
所有输出目录必须是新目录；不要覆盖旧训练集和 `glyph-mvp-v1` 模型。

## 1. 生成 V2 数据

```bash
uv run --no-sync --python .venv/bin/python poor-word glyphs generate-v2 \
  --profile mvp \
  --seed 20260911 \
  --allow-experimental \
  --output-dir data/generated/glyph-v2
```

这一步用 CPU，不占 L20。会输出持续进度和三个独立文件：

- `data/generated/glyph-v2/train.parquet`：训练，含同字的多个正常字形。
- `data/generated/glyph-v2/calibration.parquet`：只用于确定异常阈值。
- `data/generated/glyph-v2/test.parquet`：阈值固定后评估。

默认覆盖 3500 常用字；每字训练正常样本 8 张、每类异常目标 2 张；校准和测试各有正常样本 4 张、
每类异常目标 1 张。不适用的异常操作会有界重试并记录跳过，实际数量以 `run.json` 为准，
不会为了凑数回退到 V1 的弱扰动。正常样本不足会明确报错，不默默遗漏字符类别。
源文件 `data/raw/makemeahanzi_graphics.txt` 已在仓库中；缺失时按锁定来源执行
`uv run --no-sync --python .venv/bin/python poor-word data fetch --source-id makemeahanzi_graphics`。

## 2. 使用一张空闲 L20 训练

下面使用物理 **2 号卡**；若 3 号卡空闲，把 `CUDA_VISIBLE_DEVICES=2` 改为 `3`。
请先用 `nvidia-smi` 核实当前空闲卡；过去空闲不代表现在仍空闲。

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync --python .venv/bin/python poor-word train glyph \
  --manifest data/generated/glyph-v2/train.parquet \
  --sampler paired \
  --augmentation none \
  --allow-experimental \
  --epochs 20 \
  --batch-size 256 \
  --seed 20260804 \
  --pretrained \
  --device cuda \
  --log-every 25 \
  --output-dir artifacts/glyph-v2
```

这是单卡任务，不是 2/3 卡分布式训练。可见卡被映射为进程中的 `cuda:0`，不要再写 `--device cuda:2`。
每批一半正常、一半异常；正常样本按同字配对，避免之前“同字正样本很少同批出现”的问题。
批大小必须是 4 的倍数；显存不足时可改为 128。训练从 ImageNet 预训练 ConvNeXt-Tiny 开始，
不续训旧的异常检测模型。此前下载的 `convnext_tiny-983f1562.pth` 在同一用户缓存中可复用；
换用户或机器时需事先准备缓存。`--no-pretrained` 只能作为明确的从零训练对照，不能视为等价替代。

训练会打印阶段、步数和损失；结束后查看 `encoder.pt`、`prototypes.npz`、`metrics.json`。
原型只来自训练集正常样本，校准和测试样本不参与原型构建。
训练成员及文件哈希随模型保存，供后续评估验证。

## 2.1 只对训练图启用仿射增强，做 A/B 对照

复用现有 `data/generated/glyph-v2/train.parquet`、`calibration.parquet` 和 `test.parquet`；
不要重新生成数据、运行 OCR 或新增标注。先用 `nvidia-smi` 确认物理 2 号卡空闲，然后从同一预训练
权重重新开始训练（不 resume）：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync --python .venv/bin/python poor-word train glyph \
  --manifest data/generated/glyph-v2/train.parquet \
  --sampler paired \
  --augmentation affine \
  --allow-experimental \
  --epochs 20 \
  --batch-size 256 \
  --seed 20260804 \
  --pretrained \
  --device cuda \
  --log-every 25 \
  --output-dir artifacts/glyph-v2-affine
```

基线可使用参数相同的现有 `artifacts/glyph-v2`；如果其保存配置不同，则以
`--augmentation none` 重新配对训练到新目录 `artifacts/glyph-v2-control`，不能称旧结果为严格配对。
除输出目录和增强模式外，两边保持 `paired`、20 epochs、batch size 256、seed 20260804 和
pretrained 相同。增强只发生在训练取样；原型拟合、推理和评估都不增强。正常/异常样本使用相同的
参数提议分布；BLOCK 另有异常编辑可见性保护，因此实际接受率可能不同，需核对按标签记录的应用/
回退统计，不能据此保证完全没有标签线索。变换先保护字形，再由变换后的灰度图重算三视图、mask
和边缘。

策略 `glyph-affine-v1` 使用旋转 ±2°、缩放 0.97–1.03，以及每个光栅维度 ±2/128 的平移
（128 px 时为 ±2 px），最多尝试 4 次。像素级的裁切、前景拓扑及异常编辑可见性守卫拒绝不合格
候选；全部被拒时保守回退原图，并按标签记录统计。这些守卫不构成语义金标准，也不保证性能提升。

分别把 `artifacts/glyph-v2-affine` 与基线模型代入第 3 节命令，写入两个全新评估目录；例如仿射组为：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync --python .venv/bin/python poor-word evaluate glyph-v2 \
  --calibration-manifest data/generated/glyph-v2/calibration.parquet \
  --test-manifest data/generated/glyph-v2/test.parquet \
  --artifacts artifacts/glyph-v2-affine \
  --allow-experimental --max-fpr 0.0001 --prevalence 0.001 \
  --device cuda --batch-size 128 \
  --output-dir artifacts/glyph-v2-affine-eval
```

基线把 `--artifacts` 换成实际采用的 `artifacts/glyph-v2` 或 `artifacts/glyph-v2-control`，并使用
对应的全新 `artifacts/glyph-v2-baseline-eval` 输出目录。
两边都维持 `--max-fpr 0.0001`，但只能用各自校准集正常分数得到各自阈值；不要复用旧的数值阈值。
比较测试 AUROC/AUCPR、Recall、FPR、TP/FP/TN/FN 和分类型 Recall。由于这套数据已用于开发判断，
结果只是开发对照，不是新的无偏验收；相同来源的合成评估也不能证明真实业务语义或效果。

## 3. 校准阈值并评估

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync --python .venv/bin/python poor-word evaluate glyph-v2 \
  --calibration-manifest data/generated/glyph-v2/calibration.parquet \
  --test-manifest data/generated/glyph-v2/test.parquet \
  --artifacts artifacts/glyph-v2 \
  --allow-experimental \
  --max-fpr 0.0001 \
  --prevalence 0.001 \
  --device cuda \
  --batch-size 128 \
  --output-dir artifacts/glyph-v2-eval
```

只在**校准集正常样本**上确定阈值，处理并列分数后满足校准集经验 FPR 上限；
随后把同一个阈值用于测试集和各类异常，不在测试结果上重新挑阈值。
`0.0001` 表示目标 FPR 0.01%；`0.001` 表示业务异常率假设 0.1%。
校准集满足上限不保证测试集或真实业务也满足，报告会分别展示实际结果。

完成后发回 `artifacts/glyph-v2-eval/report.md` 中的阈值、测试集 AUROC/AUCPR、Recall、FPR、
TP/FP/TN/FN 和各异常类型 Recall；另附训练 `metrics.json` 的配对覆盖统计。
不需要发送公司原始图片。`scores.parquet` 保留逐样本分数，供后续错误分析。

## 4. 只诊断正常字与原型的距离

已有 `artifacts/glyph-v2` 时，可直接运行以下命令，无需重新生成数据、训练或同步依赖。
先确认物理 2 号卡空闲；`--device cuda` 仍对应可见卡映射后的 `cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync --python .venv/bin/python poor-word evaluate diagnose-prototypes \
  --train-manifest data/generated/glyph-v2/train.parquet \
  --calibration-manifest data/generated/glyph-v2/calibration.parquet \
  --artifacts artifacts/glyph-v2 \
  --characters 发纠留敞晶赣凯色法煤 \
  --allow-experimental \
  --device cuda \
  --batch-size 64 \
  --output-dir artifacts/glyph-v2-prototype-diagnostics
```

这一步冻结已有模型，以训练集所选字的 **PASS 正常样本**为原始参考，以校准集所选字的
**PASS 正常样本**为查询；校准图不会加入参考集或重新拟合原型。不读取 `test.parquet`，
不重训、不选阈值、不修改评分算法，也不改写原数据、模型、原型或旧评估产物。
输出必须是原数据及模型目录以外的**新目录**；重复运行请换一个新报告目录。

每张查询图同时比较三项余弦距离（`1 - cosine_similarity`，越小越接近）：

- `global`：到已有完整原型库的最近距离，不限于所选十字；同时报告最近原型所属字。
- `own_proto`：只到本字已有原型的最近距离。
- `own_train_nn`：只到本字训练集 PASS 原始样本嵌入的最近距离。

终端每字打印一行：选取该字 **global 分数最高的同一张校准图**，并列出它的三项距离。
不是分别取三个指标的最大值，也不是把不同图的分数拼在一起。JSON 和 Markdown 报告保留
每字全部正常查询（标准 V2 数据为 4 张），含样本标识、最近训练样本和参考数量；
先只需发回终端的 10 行摘要，不需要上传文件或抄录完整报告。

同字原型距离明显大于同字训练近邻距离，可作为原型压缩丢失局部覆盖的线索；两者都大，
则说明这张校准图在当前嵌入中也远离训练正常样本。它们只是定位证据，不是自动修复或
重新定阈值的依据；本命令不产生召回率、FPR 或模型改善结论。

## 只想先验证命令能跑通

第 1 步改用 `--profile smoke`、输出 `data/generated/glyph-v2-smoke`；第 2 步改用对应
`train.parquet`，加 `--epochs 1 --max-steps 2 --batch-size 8 --no-pretrained`，输出
`artifacts/glyph-v2-smoke`；第 3 步也换成这组小数据和模型路径，并输出新报告目录。
可以全部用 `--device cpu`，但这种两步冒烟会冻结特征层，只验证工程链路，不代表模型效果。

## 规则、标签及结论边界

- 添笔：按真实笔画尺度多笔添加；缺笔：删除整条基础笔画；断笔：单笔内部断裂，保持已确认的 2 倍切口长度。
- 位移：明确位移并与原结构重叠；粘连：保留错误封口，间隙堵塞使用已确认的加长、加粗规则。
- 正常与异常都使用相同范围的轻微外观变换，不加入只出现在某个标签上的背景特征。
- 新标签来自已校准规则，标记 `synthetic_rule_v2`，并非每张图都经人工精标；旧预览标签不自动改写。
- 三个划分不共享样本编号、生成组或完全相同的图像像素，但共享字符目录和原始字形来源。
  因此这是同来源合成实验，不是未见字体、独立业务样本或真实广告背景上的验证。
- `--allow-experimental` 只允许离线实验，不等于生产授权。Make Me a Hanzi / Arphic 来源仍为
  `production_allowed=false`，数据和模型保留实验标记；没有修改源许可或开启生产使用。
- 合成结果提升后，仍须用真实业务正常/异常样本验证误报和召回。大量同字变体不能代替足量独立真实正常图。

复现时保留 `run.json`、三份 manifest、原始笔画归档、PNG 修改声明和未修改的 `ARPHICPL.txt`；
本轮不自动将完整训练数据或新模型上传 Git。图形衍生物按所附许可提供，不提供担保。
