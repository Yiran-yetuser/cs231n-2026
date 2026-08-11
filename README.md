# CS231n Assignments

Stanford CS231n (Convolutional Neural Networks for Visual Recognition) 作业代码。
包含 assignment1 / assignment2 / assignment3。

## 快速开始

1. 按 [SETUP.md](SETUP.md) 配置 Python 环境与依赖。
2. `cd` 进对应作业目录（如 `assignments/assignment1/`），用 `cs231n (venv)` 内核启动 Jupyter：
   ```bash
   cd assignments/assignment1
   jupyter notebook
   ```
3. 每个 notebook 的第一个代码单元已适配本地环境（去除了 Colab 模板），直接运行即可。

## 内容

- **assignment1**：kNN / Softmax / Two-layer net / Features / FullyConnectedNets（numpy 实现）
- **assignment2**：PyTorch / BatchNorm / Dropout / CNN / RNN Captioning（需编译 Cython）
- **assignment3**：Transformer / 自监督 / CLIP·DINO / DDPM（需 transformers 与大模型权重）


## 目录结构

```
cs231n/
├── .venv/                    # 本地 Python 环境（忽略）
├── .gitignore
├── SETUP.md                  # 详细环境配置文档
├── download_data.sh          # curl 版数据下载脚本
├── README.md
└── assignments/
    ├── assignment1/          # kNN / Softmax / Two-layer net / Features / FullyConnectedNets
    ├── assignment2/          # 含 PyTorch / Cython 编译
    └── assignment3/          # 含 Transformers / COCO
```
