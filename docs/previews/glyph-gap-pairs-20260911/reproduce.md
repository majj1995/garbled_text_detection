# 间隙堵塞增强：5 个原位置对照

打开 [index.html](index.html) 或 [overview.png](overview.png)。每行依次是原字、旧堵塞、新堵塞。
固定用户指出的 5 个旧候选，顺序为猛、蠢、醋、派、觅；不重新抽样，不换字、笔画或连接位置。

本次 2 个位置通过（猛、觅），3 个位置跳过（蠢、醋、派）。`SKIP` 表示这个原位置不满足增强后的
检查，不能算作新生成的异常，也不会换位置补齐。2 张新候选仍为 `REVIEW`、
`training_eligible=false`，请人工判断，不自动导入标签或训练。

## 重放与变更范围

仅增强 `block_gap`：加长并加粗连接带，允许跨过原笔画两侧；在实际 96px 输入中测量
原本空白处新增的连续实心区域。旧墨迹重叠、浅色边缘、零散小点和细尾巴不能凑够强度门槛。
原有间隙拓扑、全字变化上限和主体保留检查仍然生效。
错误封口 `close_opening`、其他四类操作和已确认的原始版 2 倍断笔长度不变。

重放输入是旧候选的 `before_*` 独立笔画层，使用原种子构建原始间隙，精确匹配连接端点、
局部区域、通道跨度和笔画序号，再在该唯一位置执行新规则。通过后还要经过通用可见性检查。
`pairs.jsonl` 保存全部 5 个配对和跳过记录；`candidates.jsonl` 只含 2 个通过项。
每个新候选有 9 个视图和可重组的前后笔画 NPZ；`run.json` 保存输入及实际源码哈希。
基线是未修改的 `glyph-mixed-20260910`，其 5 个旧候选均已被用户判为合法可接受。

## 在项目根目录复现

输出目录必须不存在，使用与 `run.json` 哈希匹配的源码和依赖：

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/poor-word-uv-cache uv run --offline python \
  docs/previews/replay_gap_comparison.py \
  --baseline docs/previews/glyph-mixed-20260910 \
  --output-dir artifacts/glyph-gap-pairs-replay
```

预期输出 `sites=5 passed=2 skipped=3`。有原位置跳过时退出码为 **2**，属于明确报告的未补齐结果，
不是程序崩溃。不同生成日期会更新 PNG、NPZ 的修改声明，因此文件哈希可能改变，像素哈希可单独比较。

本批只是固定位置的强度对照，不能用于推断模型识别效果或真实图片误报率。
来源为 Make Me a Hanzi / Arphic，工程登记 `production_allowed=false`，仅用于本轮人工校准。
分享时保留 [Arphic-1999 许可全文](ARPHICPL.txt) 和各文件修改声明；不提供担保。
