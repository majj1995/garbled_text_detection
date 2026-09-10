# 单笔内部断裂：20 张专项复核

打开 [index.html](index.html)，先独立判断修改字，再展开原字、完整笔画与变化 mask。
本批全部是 `REVIEW`、`training_eligible=false`，没有训练 manifest，没有启动训练。
其他四类算子和旧预览保持不变。

## 这次修正什么

旧断笔规则可能只截短笔端、打开两笔接头，而没有切断被选中的那一笔。
新版根据独立笔画的内部骨架选点，优先转折附近，也允许足够长的笔画主体；
不按字名或左下、右上等固定方位硬编码。

- 在原分辨率与实际 96px 输入下，前景阈值 9 和核心阈值 128 都要求原笔画连续、修改后恰好分成两段。
- 两段分别检查面积、骨架像素数量（长度近似）和原笔画骨架支撑；微小碎屑不能充当第二段。
- 断口粗细使用切口附近的笔画宽度。合成全部其他笔画后，再检查局部净间距；断口被填住不能通过。
- 局部窗口里若还有不属于两侧连通区域的独立墨迹，会保守跳过。这可能降低生成率，但不以不清楚的缝隙补足配额。
- 所有几何检查都只是候选筛选，不证明汉字非法；变成另一个合法字、正常可接受写法仍应人工排除。

`candidates.jsonl` 记录切点、局部宽度、两段面积/长度近似/支撑、96px 局部净间距。
`layers/*.npz` 可重组修改前后每一笔；`run.json` 保存包括 `stroke_break.py` 在内的生成源码哈希和依赖版本。

## 固定抽样

本批种子为 `2026091029`，仅生成断笔 20 张，来自 20 个不同来源字。
排除此前四批已归档预览的 158 个来源字，3500 字池剩余 3342 字。
共尝试 28 次，跳过 8 次，没有换种子重抽、按外观挑样或用其他算子补位。
[selection.json](selection.json) 保存排除来源、哈希与字池信息。

此前 100 张的人工反馈为明显异常 94、可接受 6、不确定 0；因没有逐张编号，
没有将这 6 张推断映射到某些候选，也没有以此自动给本批打标签。

## 复现本批

使用随本批提交的代码和已锁定的笔画源，在项目根目录执行以下命令。
输出目录 `artifacts/break-preview-replay` 必须不存在。

```bash
PYTHONPATH=src uv run python - <<'PY'
import json
from pathlib import Path

from poor_word.glyphs.stroke_preview import StrokePreviewConfig, generate_stroke_preview

run = json.loads(Path("docs/previews/glyph-breaks-20260910/run.json").read_text())
config = StrokePreviewConfig.model_validate({
    **run["config"],
    "output_dir": "artifacts/break-preview-replay",
})
result = generate_stroke_preview(config, progress=lambda message: print(message, flush=True))
print(f"preview={result.html_path}")
if not result.complete:
    raise SystemExit(2)
PY
```

精确复现需匹配 `run.json` 的源码哈希与依赖版本；`selection.json` 的 `code_base_commit`
只是修改前的代码基线，不能替代实际生成版本。像素哈希、候选编号与逐样本种子可用于比对；
不同日期生成时，图片与 NPZ 内的修改日期会更新，整个文件哈希可能变化。
图形衍生物按 [Arphic-1999](ARPHICPL.txt) 提供，分享时保留许可和修改说明。

请反馈明显异常、合法可接受、不确定的数量，三项合计 20；后两类最好附候选编号。
