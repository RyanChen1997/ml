# 分类水果

import os, sys
sys.path.append(os.listdir()[2])
from PIL import Image
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


img_file_path_train = "/Users/ryan/Documents/fruit/archive/train_zip"
img_file_path_test = "/Users/ryan/Documents/fruit/archive/test_zip"
image_shape = (250, 250)
input_size = image_shape[0] * image_shape[1] * 3


# 加载数据集
def load_data(normalize=False, one_hot_label=False):
    test_data, test_label = _load_data(img_file_path_test)
    train_data, train_label = _load_data(img_file_path_train)
    print("test_data.shape:", test_data.shape)
    print("test_label.shape:", test_label.shape)
    print("train_data.shape:", train_data.shape)
    print("train_label.shape:", train_label.shape)

    if one_hot_label:
        test_label = _change_one_hot_label(test_label)
        train_label = _change_one_hot_label(train_label)

    if normalize:
        test_data = test_data.astype("float32") / 255.0
        train_data = train_data.astype("float32") / 255.0

    return (train_data, train_label), (test_data, test_label)


# 从文件中加载数据集
def _load_data(file_path):
    img_path = []
    for file in _walk_dir(file_path):
        if _is_image(file):
            img_path.append(file)

    data_list = []
    label_list = []
    for path in img_path:
        data = _read_img(path)
        if not _valid_image(data):
            continue
        data_list.append(data)
        label_list.append(_read_label(path))
    data_list = np.stack(data_list, axis=0)
    label_list = np.stack(label_list, axis=0)

    return data_list, label_list


def _read_img(file_name):
    # 读取图片
    image = Image.open(file_name)
    image = image.resize(image_shape, Image.Resampling.BILINEAR)

    # 将图片转换为numpy数组
    image_array = np.array(image)

    # 将三维数组展平为一维数组
    flat_image_array = image_array.flatten()
    return flat_image_array


# 遍历文件
def _walk_dir(file_path):
    for root, dirs, files in os.walk(img_file_path_test):
        for file in files:
            yield root + "/" + file


# 判断是否是图片
def _is_image(file_name):
    return "png" in file_name or "jpg" in file_name


def _read_label(file_name):
    if "apple" in file_name:
        return 0
    if "banana" in file_name:
        return 1
    if "orange" in file_name:
        return 2
    if "mixed" in file_name:
        return 3


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def _valid_image(data):
    return data.shape[0] / image_shape[0] / image_shape[1] == 3.0


# class TwoLayerNet:
#     def __init__(self) -> None:
#         pass

#     def gradient(self, x, t):
#         pass

#     def loss(self, x, t):
#         pass

#     def accuracy(self, x, t):
#         pass

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads



def train():
    (x_train, t_train), (x_test, t_test) = load_data(one_hot_label=True, normalize=True)

    network = TwoLayerNet(input_size=input_size, hidden_size=500, output_size=4)

    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learn_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/batch_size, 1)

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 求梯度
        grad = network.gradient(x_batch, t_batch)

        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learn_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)


if __name__ == "__main__":
    train()
