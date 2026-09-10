# 断笔切口 1× / 1.15× / 2×：固定位置对照

打开 [index.html](index.html)。本档案保留原始 20 个字符槽位，逐项展示原字、原始 1×、
既有 1.15× 和新 2×；任何倍率未通过保护检查时都留空，不重新选字、笔画或切点。

- 2× 实际通过 7 项，跳过 13 项：貌、锌、呕、蛤、煌、拎、脖、叛、疲、宝、班、佐、堰。
- 通过的 2× 中，7 项相对 1× 像素变化，
  0 项像素相同。
- 既有 1.15× 保持原档案结果：17 项通过，其中 16 项像素变化；“脖”参数变化但像素相同；3 项跳过。
- 2× 只改变沿笔画方向的切口长度；横向宽度、RNG 抽样、笔端排除、分段和可见性门槛不变。
- 全部新图均为 `REVIEW`、`training_eligible=false`；没有训练、训练 manifest 或标签继承。

## 为什么固定旧尝试

生产接口逐次提案并返回第一个合格结果。倍率改变后，更早的提案可能通过，原提案也可能失败，
所以同种子重新调用生产采样可能换笔或换位置。本离线脚本从原始 NPZ 的 `before_###` 层出发，
按原种子重放到每条记录的 `metrics.attempts`，忽略此前提案，只评估原先接受的那次尝试。
它核对来源字符、种子、笔画、切点、局部宽度和长度倍率；失败就跳过。这种固定尝试比较与生产的
“首个合格提案”采样有意不同。

## 文件与复现

`pairs.jsonl` 始终有 20 行；`candidates.jsonl` 只含通过全部保护检查的新 2× 候选。
每个新候选有 9 个视图及可重组的 `layers/*.npz` 前后笔画；旧 1×/1.15× 显示资产按原字节复制。
`run.json` 记录输入、源码与依赖哈希。修改任一输入资产会中止生成。

在项目根目录、使用匹配 `run.json` 的源码和依赖时执行：

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/poor-word-uv-cache uv run --offline python \
  docs/previews/compare_break_strengths.py \
  --baseline docs/previews/glyph-breaks-20260910 \
  --prior docs/previews/glyph-breaks-longer-20260910 \
  --output-dir artifacts/glyph-breaks-double-replay
```

预期打印 `slots=20 double_passed=7 double_changed=7 double_unchanged=0 double_skipped=13`，
并因明确存在跳过项返回退出码 2；这不是崩溃，也不会补位。日期进入 PNG/NPZ 修改说明，
跨日期生成时文件哈希可不同，但像素哈希、固定位置和通过/跳过结果应一致。

图形源自 Make Me a Hanzi / Arphic。分享时保留 [Arphic-1999 许可全文](ARPHICPL.txt)；
新图包含日期和修改说明，不提供担保。
