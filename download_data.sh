#!/bin/bash
# CS231n 数据集下载脚本（用 curl 断点续传，适配没有 wget 的 macOS）
# 用法：bash /Users/yyz/Documents/my_code/cs231n/download_data.sh
# 数据只下到 assignment1，a2/a3 已通过软链共享，无需重复下载。
set -e

DATA="/Users/yyz/Documents/my_code/cs231n/assignments/assignment1/cs231n/datasets"
mkdir -p "$DATA"
cd "$DATA"

# 如果你的网络需要代理（如 Clash / V2Ray，默认端口 7890），
# 取消下面两行注释可大幅提速：
# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=http://127.0.0.1:7890

echo "[1/2] 下载 CIFAR-10（162MB，支持断点续传，中断可重跑本脚本）..."
curl -L -C - --retry 20 --retry-delay 5 -o cifar-10-python.tar.gz \
  http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
rm -f cifar-10-python.tar.gz

echo "[2/2] 下载 imagenet_val_25.npz（features.ipynb 用）..."
curl -L -C - --retry 20 --retry-delay 5 -o imagenet_val_25.npz \
  http://cs231n.stanford.edu/imagenet_val_25.npz

echo "完成！数据已就位，assignment1/2/3 均可直接使用。"
