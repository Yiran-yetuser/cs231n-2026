# CS231n Assignments

Stanford CS231n (Convolutional Neural Networks for Visual Recognition) 作业代码。
包含 assignment1 / assignment2 / assignment3。


## 做完之后的总结

这是我第一个全部完成的计算机视觉课程作业，花了大概三四周的时间，个人感觉难度还是有的，尤其在对python代码不是特别熟练的时候。除了各个网络的架构原理之外，作业中还涉及了许多代码技巧和代码阅读理解。很多地方还是借助ai工具辅助理解完成。

### Assignment1
Assignment1主要介绍了古早的神经网络模型以及梯度下降推导，实现的模型较为朴素和易解，应该可以算是学习ai的通识基础。  
- kNN: k-Nearest Neighbors，最邻近算法，主要是通过计算样本之间的距离来进行分类。这个算法并不强大，但是是一个很好的入门算法，帮助理解分类问题的基本概念和基本思想。同时熟悉了numpy的使用。
- Softmax: Softmax分类器，主要是通过softmax函数将输出转化为概率分布，从而进行多分类问题的处理。这个算法在深度学习中非常常用。难点是实现梯度下降算法。当时在做的时候不熟悉梯度下降的推导，在高铁上做了3h+ : (
- Two-layer net: 两层神经网络。不复杂的实现，主要是通过两层神经网络来进行分类问题的处理。相当于介绍了神经网络的基本结构和前向传播、反向传播的基本原理。
- Features: 特征提取。主要是通过提取图像的特征来进行分类问题的处理。这个算法在计算机视觉中非常常用，后面都是它的身影。感觉是很有趣的一个作业，不过好像其实基本不需要做什么事情（）
- FullyConnectedNets: 全连接神经网络。主要是通过全连接神经网络来进行分类问题的处理。这个算法感觉才正式进入了深度学习的范畴，虽然单拿出来不是一个强大的神经网络算法，但是在后面的学习还是经常用到。

### Assignment2
Assignment2主要介绍了卷积神经网络（CNN）以及优化方法。
- CNN: 超经典。卷积神经网络。主要是通过卷积操作来提取图像的特征，从而进行分类问题的处理。之前早有耳闻，这次真手动实现了（一部分内容）
- Dropout: 正则化方法。主要是通过随机丢弃神经元来防止过拟合，从而提高模型的泛化能力。
- Batch Normalization: 批量归一化。主要是通过对每一层的输入进行归一化处理。这个相对好理解一点，但是学到Layer Normalization的时候就感觉有点吃力。
- RNN: 循环神经网络。不过这里的实现主要是Captioning，生成图片的描述。RNN实际上就是在CNN的基础上加了一个循环结构，能够处理序列数据，并不是很难的实现。
- PyTorch: 学习pytorch

### Assignment3
真正开始实现一些复杂的模型。
- Transformers: 主要是通过自注意力机制来处理序列数据，从而进行分类问题的处理。在这份作业里主要是实现了Multi-Head Attention\Transformer Encoder\ViT。Attention机制理解起来不简单。
- Self-Supervised_Learning: 自监督学习。感觉相当美妙，通过对比学习就可以让模型学习到数据的特征表示，从而进行分类问题的处理。我对这里面的数学原理还是很感兴趣的。
- CLIP: 主要是通过对图像和文本进行联合训练，从而实现图像和文本的相互理解。自监督学习的一个应用，多模态。
- DINO：对比学习，自蒸馏方法。学生网络配合教师动量网络，拥有强大的泛化性能！和CLIP类似，都是自监督学习的应用。与ViT结合，有强大的特征提取能力！
- DDPM：传奇扩散模型，杀死了GAN，生成模型的未来。不过它的数学原理我还不是很理解，尤其是和马尔可夫相关的部分。和Unet结合。

## 快速开始

1. 按 [SETUP.md](SETUP.md) 配置 Python 环境与依赖。
2. `cd` 进对应作业目录（如 `assignments/assignment1/`），用 `cs231n (venv)` 内核启动 Jupyter：
   ```bash
   cd assignments/assignment1
   jupyter notebook
   ```
3. 每个 notebook 的第一个代码单元已适配本地环境（去除了 Colab 模板），直接运行即可。

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
