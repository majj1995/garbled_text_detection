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
