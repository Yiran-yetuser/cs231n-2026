# CS231n Assignments

Stanford CS231n (Convolutional Neural Networks for Visual Recognition) 作业代码。
包含 assignment1 / assignment2 / assignment3。

---

## 环境说明

本仓库使用 **本地 Python 虚拟环境（venv）**，不依赖 conda。

- Python 版本：3.13.12（受管运行时）
- 虚拟环境位置：仓库根目录下的 `.venv/`（**不进 git**）
- Jupyter 内核名：`cs231n (venv)`（已注册到本机用户）

### 在本地打开作业

```bash
# 1. 进入具体作业目录（必须在 assignmentX 内启动，否则 import cs231n 失败）
cd /Users/yyz/Documents/my_code/cs231n/assignments/assignment1

# 2. 激活环境
source /Users/yyz/Documents/my_code/cs231n/.venv/bin/activate

# 3. 启动 Jupyter（浏览器自动打开，内核选 cs231n (venv)）
jupyter notebook
```

### 依赖清单（已在 .venv 中安装并验证）

| 类别 | 包 |
|------|-----|
| 科学计算 | numpy, scipy, matplotlib |
| Notebook | jupyter, notebook, ipykernel |
| 深度学习 | torch, torchvision, torchaudio（macOS MPS 加速可用） |
| A3 额外 | transformers, Pillow, imageio |
| 编译 | Cython（用于 A2 扩展） |

> 注：不要照抄 `assignment3/requirements.txt`，它钉死的是 2020 年前后的旧版本，在 Python 3.13 上装不上。直接使用上面的现代版本即可。

---

## 关键注意事项（避坑）

1. **必须在作业目录下启动 Jupyter**
   每个 `assignmentX/` 里都有本地包 `cs231n/`，notebook 靠 `import cs231n` 引用它。`cd` 进 `assignment1/`（或 2、3）再开 Jupyter，否则会 `ModuleNotFoundError`。

2. **assignment2 需要编译 Cython 扩展**
   ```bash
   cd assignments/assignment2/cs231n
   pip install cython
   python setup.py build_ext --inplace
   ```
   成功后生成 `im2col_cython.cpython-313-darwin.so`。**该 .so 是平台相关的，不进 git，换环境（如 Colab）需重新编译。**

3. **数据集不进 git**
   `cifar-10-batches-py/`、`imagenet_val_25.npz` 等数据文件被 `.gitignore` 排除。数据已下载到 `assignment1/cs231n/datasets/`，并通过软链共享给 a2/a3。重新 clone 后需重新下载（见下）。

4. **Colab 专属代码不要留在本地 notebook 里**
   若从 Colab 模板复制，`from google.colab import drive`、`!bash get_datasets.sh`（依赖 wget，macOS 默认没有）这类代码请在本地删除或注释掉。

5. **官方 `get_datasets.sh` 在 macOS 跑不通**
   它内部用 `wget`，而 macOS 默认无 wget。改用仓库根目录的 `download_data.sh`（curl 版，支持断点续传）。

---

## 数据集

| 数据集 | 用途 | 下载方式 |
|--------|------|---------|
| CIFAR-10 (`cifar-10-batches-py/`) | a1 / a2 / a3 核心 | `bash download_data.sh` 或 Colab 下载后拷回 |
| imagenet_val_25.npz | 可视化 / features | 同上，脚本一并下载 |
| COCO captioning (`coco_captioning/`) | a3 的 Captioning / CLIP 题 | `cs231n/datasets/get_coco_dataset.sh`（Colab 有 wget 可用；本地需改 curl） |
| emoji_data.npz / text_embeddings.pt | a3 的 emoji / transformer 题 | 运行时自动下载，无需手动 |

数据存放位置：`assignments/assignment1/cs231n/datasets/`（a2、a3 经软链共享）。

---

## 工作流（本地 + Colab + GitHub）

- **GitHub**：只存代码和改动的 notebook，环境(`.venv`)、数据(`datasets`)、编译产物(`*.so`)均被 `.gitignore` 忽略。
- **本地 (Mac)**：用 venv 跑日常开发与调试。
- **Colab**：借云端 GPU 跑重训练。clone 后需重新编译 Cython 扩展、重新下载数据（Colab 到官方源速度远快于本机）。

### 首次提交到 GitHub

```bash
cd /Users/yyz/Documents/my_code/cs231n
git add .
git status        # 确认没有 .venv / datasets / .so 被加入
git commit -m "cs231n: init assignments"
git remote add origin git@github.com:你的用户名/cs231n.git
git branch -M main
git push -u origin main
```

---

## 目录结构

```
cs231n/
├── .venv/                    # 本地 Python 环境（忽略）
├── .gitignore
├── SETUP.md                  # 详细环境配置文档
├── download_data.sh          # curl 版数据下载脚本
├── README.md
└── assignments/
    ├── assignment1/          # kNN / SVM / Softmax / Two-layer net / Features
    ├── assignment2/          # 含 PyTorch / Cython 编译
    └── assignment3/          # 含 Transformers / COCO
```
