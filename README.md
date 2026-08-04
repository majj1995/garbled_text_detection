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
