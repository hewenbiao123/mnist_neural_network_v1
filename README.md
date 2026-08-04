# MNIST 手写数字识别 - 纯 NumPy 神经网络

基于纯 NumPy 实现的前馈神经网络，用于 MNIST 手写数字识别。不使用 TensorFlow、PyTorch 等深度学习框架，仅依赖 NumPy 从零搭建神经网络，适合理解神经网络底层原理。

## 网络架构

| 层 | 神经元数 | 激活函数 |
|---|---|---|
| 输入层 | 784（28×28 像素展平） | - |
| 隐藏层 1 | 100 | Sigmoid |
| 隐藏层 2 | 50 | Sigmoid |
| 输出层 | 10（对应数字 0-9） | Softmax |

## 项目结构

```
mnist_neural_network/
├── neural_network.py          # 主入口：加载权重，执行推理
├── sample_weight.pkl          # 预训练好的模型权重
├── common/
│   └── functions.py           # 激活函数：sigmoid、relu、softmax
├── dataset/
│   ├── mnist.py               # MNIST 数据集加载器（下载、解析、持久化）
│   └── mnist.pkl              # 序列化后的 MNIST 数据集（NumPy 格式）
└── README.md
```

## 环境依赖

- Python 3.8+
- NumPy
- Pillow（仅用于 `_test()` 图片展示函数，非必需）

安装依赖：

```bash
pip install numpy pillow
```

## 快速开始

```bash
cd nist_neural_network_v1
python neural_network.py
```

首次运行时，程序会自动从网络下载 MNIST 数据集（约 11MB）并转换为 NumPy 格式缓存到本地，之后运行将直接加载缓存文件。

## 预期结果

```
Accuracy: 0.9352
```

预训练模型在 MNIST 测试集上达到约 **93.52%** 的准确率。

## 主要特性

- **纯 NumPy 实现**：前向传播完全基于 NumPy 矩阵运算，无深度学习框架依赖
- **自动数据集管理**：自动下载、解析 MNIST 数据集并持久化缓存，无需手动准备数据
- **批量推理**：支持批量（batch）处理，提高推理效率
- **模块化设计**：激活函数、数据集加载、网络推理分离，结构清晰

## 注意事项

- 本项目仅包含**推理（inference）**代码，训练代码不在本项目中。`sample_weight.pkl` 是已经训练好的权重文件
- 数据集首次下载需要网络连接，下载地址为 `https://ossci-datasets.s3.amazonaws.com/mnist/`
- 数据集文件（约 11MB × 4）和缓存文件（`mnist.pkl`）会保存在 `dataset/` 目录下，首次运行会自动下载
