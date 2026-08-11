# 环境配置（Setup）

仅说明本仓库的 Python 环境如何搭建。

## 1. 创建虚拟环境

推荐 Python 3.13

##  依赖清单

| 类别 | 包 |
|------|-----|
| 科学计算 | numpy, scipy, matplotlib |
| Notebook | jupyter, notebook, ipykernel |
| 深度学习 | torch, torchvision, torchaudio（macOS MPS 加速可用） |
| A3 额外 | transformers, Pillow, imageio |
| 编译 | Cython（用于 A2 扩展） |

> 不要照抄 `assignment3/requirements.txt`，它钉死的是 2020 年前后的旧版本，在 Python 3.13 上装不上。直接使用上表的现代版本即可。

安装：

```bash
pip install numpy scipy matplotlib jupyter notebook ipykernel \
            torch torchvision torchaudio transformers Pillow imageio Cython
```

## 3. 注册 Jupyter 内核

```bash
python -m ipykernel install --user --name cs231n --display-name "cs231n (venv)"
```

## 4. A2 编译前置（Cython）

assignment2 的卷积扩展需要本地编译一次：

```bash
cd assignments/assignment2/cs231n
pip install cython
python setup.py build_ext --inplace
```

生成 `im2col_cython.cpython-313-darwin.so`（平台相关，不进 git，换环境需重编）。
