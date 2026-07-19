# Pinenut

[![PyPI](https://img.shields.io/pypi/v/pinenut.svg)](https://pypi.org/project/pinenut/)
[![Python](https://img.shields.io/pypi/pyversions/pinenut.svg)](https://pypi.org/project/pinenut/)
[![Tests](https://github.com/pinenuthome/pinenut/actions/workflows/tests.yml/badge.svg)](https://github.com/pinenuthome/pinenut/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pinenuthome/pinenut/blob/main/LICENSE)

[English](https://github.com/pinenuthome/pinenut/blob/main/README.md) | 简体中文

Pinenut 是一个使用 Python 编写的轻量级动态计算图深度学习框架。它提供自动
微分、神经网络模块、优化算法，以及统一的 NumPy/CuPy API，可在 CPU 和
NVIDIA GPU 上运行。项目代码量小，适合阅读、实验和扩展。

## 主要特性

- 动态计算图以及一阶、高阶梯度
- 统一的 `Module`、`Parameter` 和 `Optimizer` 接口
- 常用网络层、激活函数、损失函数、数据集和优化器
- 使用 NumPy 在 CPU 上运行，可选用 CuPy 加速 NVIDIA GPU
- 支持模型状态字典以及同步多 GPU 训练
- 纯 Python 实现，适合学习原理和小型实验

## 环境要求

- Python 3.8 或更高版本
- NumPy 1.20 或更高版本
- 可选 GPU 支持：兼容的 NVIDIA 驱动，以及一个受支持的 CuPy 包

## 安装

从 PyPI 安装最新版本：

```bash
python -m pip install --upgrade pinenut
```

也可以克隆仓库，以可编辑模式安装当前源码：

```bash
git clone https://github.com/pinenuthome/pinenut.git
cd pinenut
python -m pip install -e .
```

如需使用 NVIDIA GPU，请根据 CUDA 主版本选择对应的扩展依赖：

```bash
# CUDA 12.x
python -m pip install "pinenut[gpu-cu12]"

# CUDA 13.x
python -m pip install "pinenut[gpu-cu13]"
```

同一个环境中只能安装一种 CuPy 发行包。有关 Linux 虚拟环境、不同发行版的
系统依赖、CUDA 故障排查和真实双 GPU 验证，请参阅
[Linux 安装指南](https://github.com/pinenuthome/pinenut/blob/main/docs/linux-installation.md)。

## 自动微分

创建动态计算图，并对标量结果求导：

```python
import numpy as np

import pinenut as pn

x = pn.tensor(np.array([1.0, 2.0, 3.0]))
loss = pn.sum(x ** 2)
loss.backward()

print(loss.data)    # 14.0
print(x.grad.data)  # [2. 4. 6.]
```

如果后续计算还需要对梯度求导，请设置 `create_graph=True`：

```python
x = pn.tensor(3.0)
y = x ** 2
y.backward(create_graph=True)

first_derivative = x.grad
x.zero_grad()
first_derivative.backward()
print(x.grad.data)  # 2.0
```

## 神经网络训练

模块向优化器提供参数，并通过明确的训练和评估模式控制行为：

```python
import numpy as np

import pinenut as pn
from pinenut import nn, optim
from pinenut.nn import functional as F

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 3),
)
optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = pn.tensor(np.random.randn(8, 2))
targets = pn.tensor(np.random.randint(0, 3, size=8))

model.train()
optimizer.zero_grad()
logits = model(inputs)
loss = F.cross_entropy(logits, targets)
loss.backward()
optimizer.step()

model.eval()
with pn.no_grad():
    predictions = model(inputs)
```

当前提供 `Linear`、`LazyLinear`、`Sequential`、`MLP`、`Embedding`、
`Dropout`、`SGD`、`Adagrad` 和 `Adam`。`SGD` 可以设置可选的 `momentum`。
分类损失函数接收未经归一化的 logits；使用 `F.cross_entropy` 时，最后一层不应
再应用 softmax。

## 设备

同一套模型和张量 API 可以在 CPU 与 GPU 上使用：

```python
device = 'cuda' if pn.cuda.is_available() else 'cpu'
model.to(device)
inputs.to(device)
targets.to(device)
```

可以使用 `cuda:1` 这样的名称选择特定 GPU。调用 `cpu()`、`cuda()` 或
`to(device)` 会就地移动张量或模块，并返回其自身。

## 保存模型状态

```python
pn.save(model.state_dict(), 'model_weights.npz')

restored = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 3),
)
restored.load_state_dict(pn.load('model_weights.npz'))
restored.eval()
```

## 示例

- [MNIST 训练](https://github.com/pinenuthome/pinenut/blob/main/examples/mnist/train_mnist.py)
- [MNIST 多 GPU 训练](https://github.com/pinenuthome/pinenut/blob/main/examples/mnist/train_mnist_gpus.py)
- [螺旋数据分类](https://github.com/pinenuthome/pinenut/blob/main/examples/nn/test_spiral.py)
- [Embedding 示例](https://github.com/pinenuthome/pinenut/blob/main/examples/nn/test_embedding.py)
- [Dropout 示例](https://github.com/pinenuthome/pinenut/blob/main/examples/nn/test_dropout.py)

运行 Spiral 可视化 Demo，训练 MLP 并保存决策边界、训练指标和状态字典：

```bash
python examples/nn/test_spiral.py
python examples/nn/test_spiral.py --animate
```

输出文件默认保存在 `demo_outputs/spiral/`。运行 `--help` 可以配置数据集、
模型、训练轮数、输出目录以及 CPU/GPU 设备。

## 测试

安装开发依赖，然后运行兼容 CPU 的测试套件：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m unittest discover -s test -v
```

在配备两张物理 NVIDIA GPU 的机器上执行多 GPU 冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/test_multigpu.py --devices 0,1
```

如果要测试物理 GPU 2 和 3，可将它们映射为逻辑设备 0 和 1：

```bash
CUDA_VISIBLE_DEVICES=2,3 python tools/test_multigpu.py --devices 0,1
```

当设备分配、训练、参数同步和有限值检查全部通过时，命令会输出
`MULTI-GPU SMOKE TEST PASSED`。

## 项目结构

```text
core/       自动微分与内部实现
nn/         神经网络模块与无状态函数
optim/      优化算法
datasets/   内置数据集
utils/      数据加载与工具接口
examples/   可运行的训练与可视化示例
test/       单元测试与回归测试
tools/      独立验证工具
docs/       安装与使用文档
```

## 参与贡献

欢迎提交 Issue 和 Pull Request。对于影响行为的改动，请同步新增或更新测试，
并在提交前运行完整测试套件。

## 许可证

Pinenut 基于
[MIT License](https://github.com/pinenuthome/pinenut/blob/main/LICENSE) 发布。
