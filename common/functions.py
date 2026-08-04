"""
定义常用激活函数
"""

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    max_value = np.max(x)
    exp_x = np.exp(x - max_value)
    return exp_x / np.sum(exp_x)