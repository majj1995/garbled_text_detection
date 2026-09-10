# 五类规则冻结后的 100 张混合复核

打开 [index.html](index.html)，先独立判断修改字，再展开原字、完整笔画和变化 mask。
[overview.png](overview.png) 与审阅模板使用相同的固定打乱顺序，编号可用于反馈。
本批只调用已有生成器，不修改算法，不训练模型；全部为 `REVIEW`、`training_eligible=false`。

## 已确认的范围

用户已确认断笔切除长度采用原始版的 2 倍；切口横向宽度、笔画两段保留和局部净空等保护规则不变。
添笔、整笔删除、粘连、重叠位移沿用当前代码。强度确认不等于本批或旧批的逐图异常标签。

## 固定抽样与实际结果

- 生成代码基线：`ee04742b62b657843f438055df2c104d327f1709`；实际源码和依赖哈希见 `run.json`。
- 种子：`2026091053`；五类各 20 张，槽位不在类型间相互补齐。
- 排除此前七个预览档案中全部 178 个来源字；3500 常用字池剩余 3322 字。
- 沿用有放回的随机抽样和像素去重规则，本次实际生成 100 张、来自 100 个不同汉字。
- 共尝试 112 次，12 次因不适用而跳过，没有缺额；跳过记录见 `run.json`。
- 粘连实际包含错误封口 8 张、间隙堵塞 12 张；本次未另设子型配额。
- 未按生成后的外观挑样，未换种子重抽；展示顺序使用生成器的固定打乱规则。

[selection.json](selection.json) 保存旧候选清单的哈希、排除字、字池哈希和抽样结果。
本批是新字上的规则质量抽检，不是上一批同位置的长度对照，也不用于模型准确率或现网误报率评估。

## 在项目根目录复现

先按项目说明准备 uv 环境及已锁定的笔画源和许可，使用与本批 `run.json` 匹配的源码和依赖。
下面的输出目录必须不存在；源预览目录保持不动。

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/poor-word-uv-cache uv run --offline python - <<'PY'
import json
from pathlib import Path

from poor_word.glyphs.stroke_preview import StrokePreviewConfig, generate_stroke_preview

run = json.loads(Path("docs/previews/glyph-mixed-20260910/run.json").read_text())
config = StrokePreviewConfig.model_validate({
    **run["config"],
    "output_dir": "artifacts/glyph-mixed-replay",
})
result = generate_stroke_preview(config, progress=lambda message: print(message, flush=True))
print(f"preview={result.html_path} count={result.candidate_count} complete={result.complete}")
if not result.complete:
    raise SystemExit(2)
PY
```

匹配版本时应得到 100/100、112 次尝试、12 次跳过。候选编号、逐样本种子和像素哈希可用于比对；
不同日期生成时，PNG 和 NPZ 中的日期声明会更新，整个文件 SHA-256 可能不同。
每个候选有 9 个视图，`layers/*.npz` 保存可重组的修改前后独立笔画。

## 反馈与使用范围

每类分别汇总明显异常、合法可接受、不确定，三项合计 20；后两项尽量附候选编号。
变成另一个合法字不属于乱码；还能猜出原字不等于字形合法。
本轮不自动导入标签，也不生成训练 manifest。

图形源自 Make Me a Hanzi / Arphic，资源在工程中登记为 `production_allowed=false`，仅用于本轮人工校准。
本批不包含复杂广告背景或多字体迁移，不应直接视为正式生产训练集。
分享时保留 [Arphic-1999 许可全文](ARPHICPL.txt) 及图片、NPZ 中的修改声明；不提供担保。
