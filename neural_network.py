"""
此处假设神经网络已经训练好了, 拥有了模型权重: sample_weight.pkl
该神经网络架构如下：
1. 输入层: 784 个神经元，对应一张图片 28 * 28 个像素点
2. 隐藏层1: 100 个神经元，使用 sigmoid 激活函数
3. 隐藏层2: 50 个神经元，使用 sigmoid 激活函数
4. 输出层: 10 个神经元，对应一张图片代表数字(0-9)
"""

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import os

weight_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_weight.pkl")

def get_data() -> tuple[np.ndarray, np.ndarray]:
    (x_train, t_train), (x_test, t_test) = load_mnist()
    return x_test, t_test

def init_network():
    with open(weight_file_path, 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# def run_network():
#     x, t = get_data()
#     network = init_network()
#     accuracy_cnt = 0
#     for i in range(len(x)):
#         y = predict(network, x[i])
#         p = np.argmax(y)    # 取最大值对应的索引
#         accuracy_cnt += 1 if p == t[i] else 0
#     print(f"Accuracy: {float(accuracy_cnt) / len(x)}") # Accuracy: 0.9352

def run_network(batch_size=1):
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        # axis: 想象成一个移动的指针，即“请沿着这个方向去聚合数据”
        p = np.argmax(y_batch, axis=1) # 按照y轴取最大值索引
        accuracy_cnt += np.sum(p==t[i:i+batch_size])
    print(f"Accuracy: {float(accuracy_cnt) / len(x)}")



run_network(batch_size=100) 
