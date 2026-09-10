# 五类扩样复核：抽样与复现

本批只使用现有笔画生成规则，不更改算法，不训练模型。
请打开 [index.html](index.html) 先判断修改字，再展开原字和修改证据。
全部候选仍为 `REVIEW`、`training_eligible=false`，不能直接当作已确认异常训练集。

## 固定抽样

- 代码基线：`01da55a846eaa43432677ab1424a89e48a349222`；各生成模块的实际 SHA-256 保存在 `run.json`。
- 批次种子：`2026091017`；每类 20 张，不按生成后的外观挑样或换种子重抽。
- 排除 `glyph-v2-20260910`、`glyph-strokes-20260910`、`glyph-bridges-20260910` 中的全部 59 个来源字；3500 字池剩余 3441 字。
- 按现有规则有放回抽取来源字，候选像素去重：100 张对应 99 个来源字，“落”出现于两种不同操作。
- 五类均为 20 张；粘连的实际子型为错误封口 11 张、间隙堵塞 9 张，本批未设子型配额。
- 共尝试 131 次，31 次不适用被跳过，未出现配额缺失；具体原因见 `run.json`。

[selection.json](selection.json) 保存旧候选清单的哈希、排除字和新字池哈希。
旧预览保持原样；此前粘连 20 张的人工反馈是明显异常 20、合法可接受 0、不确定 0，
这一反馈没有用于推断本批标签，也没有自动导入训练。

## 在项目根目录复现

先按项目说明安装 uv 环境，并保留已锁定的笔画源和许可。以下命令从已归档配置复现，
输出到新的 `artifacts/glyph-expanded-replay` 目录；该目录必须不存在。

```bash
PYTHONPATH=src uv run python - <<'PY'
import json
from pathlib import Path

from poor_word.glyphs.stroke_preview import StrokePreviewConfig, generate_stroke_preview

run = json.loads(Path("docs/previews/glyph-expanded-20260910/run.json").read_text())
config = StrokePreviewConfig.model_validate({
    **run["config"],
    "output_dir": "artifacts/glyph-expanded-replay",
})
result = generate_stroke_preview(config, progress=lambda message: print(message, flush=True))
print(f"preview={result.html_path}")
if not result.complete:
    raise SystemExit(2)
PY
```

精确复现需保持上述代码及 `run.json` 中的依赖版本。候选像素哈希、编号与逐样本种子可用于比对；
不同日期运行时，图片和归档中的日期声明会更新，因此不能要求整个文件 SHA-256 不变。
图形衍生物使用 [Arphic-1999](ARPHICPL.txt) 许可，分享时保留许可与修改声明。

## 反馈方式

每类 20 张，分别汇总明显异常、合法可接受、不确定，三项合计应为 20。
优先附上可接受或不确定样本的编号；变成另一个合法字不算乱码。
这轮判断的是候选字形质量，不是模型识别准确率，也不验证复杂广告背景下的效果。
