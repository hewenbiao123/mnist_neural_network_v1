"""
导入 mnist 数据集
1. 下载 MNIST 数据集
2. 读取二进制数据, 将数据转化为numpy数组
3. 数据持久化保存在本地
"""

import urllib.request   # 网络请求库 
import gzip             # 用于读取.gz压缩文件
import pickle           # 将数据持久化保存到磁盘
import os.path          # 用于处理文件路径
import numpy as np
from typing import TypeAlias # 用于类型别名

# ===================== 下载 MNIST 数据集 =====================

# 下载地址
download_url: str = "https://ossci-datasets.s3.amazonaws.com/mnist/" 
# 下载的文件
download_file: dict[str, str] = {       
    'train_img':'train-images-idx3-ubyte.gz',   # 训练图片
    'train_label':'train-labels-idx1-ubyte.gz', # 训练标签
    'test_img':'t10k-images-idx3-ubyte.gz',     # 测试图片
    'test_label':'t10k-labels-idx1-ubyte.gz'    # 测试标签
}

# 下载文件存储目录（默认为当前文件所在目录）
save_dir = os.path.dirname(os.path.abspath(__file__))
# 持久化数据保存路径
pickle_file_path = os.path.join(save_dir, "mnist.pkl")

# 步骤一：下载 MNIST数据集
def download_mnist():
    for file_name in download_file.values():
        save_file_path = os.path.join(save_dir, file_name)
        # 如果已下载，不执行下载
        if os.path.exists(save_file_path):
            continue
        print(f"Downloading {file_name} ...")
        # 参数(下载地址, 保存路径)
        urllib.request.urlretrieve(download_url + file_name, save_file_path)
        print("Done!")


# ===================== 读取 MNIST 数据集，转化为 Numpy 数组格式 =====================

image_size = 28 * 28

# 读取标签数据，转化为 numpy 数组格式
def _load_label(file_name: str) -> np.ndarray:
    file_path = os.path.join(save_dir, file_name)
    print(f"Converting {file_name} to Numpy Array ...")
    with gzip.open(file_path, 'rb') as f:
        # 从二进制数据中读取标签，offset=8 表示跳过 8 个字节
        # 标签值是0-9，使用 uint8 读取，对应 numpy 的 dtype 是 uint8
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done!")
    return labels

# 读取图片数据，转化为 numpy 数组格式
def _load_img(file_name: str) -> np.ndarray:
    file_path = os.path.join(save_dir, file_name)
    print(f"Converting {file_name} to Numpy Array ...")
    with gzip.open(file_path, 'rb') as f:
        # 从二进制数据中读取图片，offset=16 表示跳过开头 16 个字节
        # 一个像素点值为0-255，使用 uint8 读取，对应 numpy 的 dtype 是 uint8
        pixels = np.frombuffer(f.read(), np.uint8, offset=16)
    # 一张图片有 28 * 28 个像素点，按照 28 * 28 为一行
    images = pixels.reshape(-1, image_size)
    print("Done!")
    return images 

# 步骤二：读取数据，转化为 numpy 数组
def convert_numpy() -> dict[str, np.ndarray]:
    dataset: dict[str, np.ndarray] = {}
    for name, file_name in download_file.items():
        dataset[name] = (
            _load_img(file_name) if name.endswith('img') else _load_label(file_name)
        )
    return dataset

# ===================== 持久化存储 MNIST 数据集 Numpy 格式 =====================

# 步骤三：持久化存储到本地
def save_as_pickle(dataset: dict[str, np.ndarray]):
    print("Creating pickle file ...")
    with open(pickle_file_path, 'wb') as f:
        # 将 mnist 的 Numpy 格式数据序列化到本地
        pickle.dump(dataset, f, -1) # -1表示使用最高版本的序列化协议
    print("Done!")



def init_mnist():
    download_mnist()
    dataset = convert_numpy()
    save_as_pickle(dataset)


ImagePair: TypeAlias = tuple[np.ndarray, np.ndarray]
label_max_value = 9

# API，支持将标签数据独热编码，将图像像素值正规化为0.0-1.0
def load_mnist(normalize=True, flatten=True, one_hot_label=False) -> tuple[ImagePair, ImagePair]:
    """
    读入 MNIST 数据集

    Parameters
    ----------
        normalize: bool : 是否将图像像素值正规化为0.0-1.0
        flatten: bool : 是否将图像展开为一维数组（表示一张图像用一维数组表示）
        one-hot: bool : 是否对标签数据one-hot编码

    Returns
    -------
        tuple[ImagePair, ImagePair]: (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(pickle_file_path):
        init_mnist()

    with open(pickle_file_path, 'rb') as f:
        dataset = pickle.load(f)    # pickle 文件反序列化

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32) # 转化数组类型
            dataset[key] /= 255.0

    if one_hot_label:
        for key in ('train_label', 'test_label'):
            T = np.zeros((dataset[key].size, label_max_value + 1)) # 创建 n * 10 的全零矩阵
            for idx in range(len(T)):
                T[idx][dataset[key][idx]] = 1
            dataset[key] = T

    # 将一张图像转化为三维张量，(灰度, 行像素数, 列像素数)
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    train_pair = (dataset['train_img'], dataset['train_label'])
    test_pair = (dataset['test_img'], dataset['test_label'])
    return (train_pair, test_pair)

# 测试函数，查看图片
def _test(idx: int):
    (train_images, train_labels), (test_images, test_labels) = load_mnist(normalize=False)
    image = train_images[idx].reshape(28, 28) # 转化为图片原本尺寸
    label = train_labels[idx]
    print(f"The label of image is {label}.")
    print(f"The shape of image is {image.shape}.")
    from PIL import Image
    Image.fromarray(np.uint8(image)).show()

if __name__ == '__main__':
    init_mnist()
    # _test(2)


