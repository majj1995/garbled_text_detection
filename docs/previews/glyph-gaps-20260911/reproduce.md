# 间隙堵塞加强版：20 张新字专项复核

打开 [index.html](index.html)，先看 96px 修改字，再展开原字、涉及笔画和变化 mask。
[overview.png](overview.png) 与审阅模板的顺序一致。本批只包含 `bridge:block_gap`，
用于判断加长、加粗后的连接是否仍像可接受的噪点；不包含错误封口或其他四类操作。

## 固定抽样与结果

- 预改动代码基线 `4714194f62b093a0c025ffcf0f5640cc56a1a284`；实际生成源码和依赖以 `run.json` 哈希为准。
- 固定种子 `2026091107`，单子型配额 20；每个槽位最多 48 次尝试，不跨子型补齐。
- 排除前八个归档中全部 278 个来源字，按原始常用字目录顺序保留剩余 3222 字。
- 本次得到 20 张、来自 20 个不同汉字，共尝试 209 次、跳过 189 次，无缺额。
- 沿用有放回抽样、像素去重和固定打乱展示；未按外观挑样、删掉已生成候选或换种子重抽。

[selection.json](selection.json) 保存字池和排除清单哈希、抽样策略与旧批 95 / 5 / 0 的用户反馈。
`run.json` 保存全部跳过记录。较高跳过数代表几何和可见性检查更严格，不是异常率或模型召回率。
每个候选有 9 个视图和前后独立笔画 NPZ。

## 改动边界

连接带厚度为局部笔画宽度的 2 倍，中心线长度包含间隙跨度及两侧延伸；允许明显超出已有笔画。
仍限制全字尺度、画布边界和总变化比例，避免整字被覆盖。
实际 96px 输入中，只有原本空白且新增为实心的区域参与强度检查；经 3×3 开运算去除细尾巴，
再检查单个连通块的面积、最大边界框跨度和内部半径，不能靠多个散点累加通过。
原间隙的连通性与吞并比例检查仍然保留。这些几何证据不能证明汉字非法。

错误封口、其他四类操作和已确认的原始版 2 倍断笔长度均未修改。
需要观察同一原字、同一位置的变化时，请看 [5 个旧位置对照](../glyph-gap-pairs-20260911/index.html)，
不要把本批新字与旧批不同字直接当作逐样本强度对比。

## 在项目根目录复现

使用与 `run.json` 匹配的源码和依赖，输出目录必须不存在：

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/poor-word-uv-cache uv run --offline python - <<'PY'
import json
from pathlib import Path
from poor_word.glyphs.stroke_preview import StrokePreviewConfig, generate_stroke_preview

run = json.loads(Path("docs/previews/glyph-gaps-20260911/run.json").read_text())
config = StrokePreviewConfig.model_validate({
    **run["config"],
    "output_dir": "artifacts/glyph-gaps-replay",
})
result = generate_stroke_preview(config, progress=lambda message: print(message, flush=True))
print(f"preview={result.html_path} count={result.candidate_count} complete={result.complete}")
if not result.complete:
    raise SystemExit(2)
PY
```

预期 20/20、209 次尝试、189 次跳过。候选编号、种子及像素哈希可用于核对；
不同日期产生的 PNG、NPZ 修改声明会变化，文件哈希因此可能不同。
CLI 同样支持 `glyphs preview-strokes --bridges-only --bridge-mode block_gap --per-operator 20`，
但还需提供相同的来源、许可、字池和种子；单独使用 `--bridges-only` 会生成两个子型各 20 张。

## 人工反馈与使用范围

请汇总明显异常 / 合法可接受 / 不确定，合计 20；后两项附候选编号。
变成另一个合法字不属于乱码；还能猜出原字不等于字形合法。
全部候选保持 `REVIEW`、`training_eligible=false`，不导入标签、不生成训练 manifest、不启动训练。
本批不含复杂广告背景，也不评估模型或现网误报率。

来源为 Make Me a Hanzi / Arphic，工程登记 `production_allowed=false`，仅用于本轮人工校准。
分享时保留 [Arphic-1999 许可全文](ARPHICPL.txt) 和修改声明；不提供担保。
