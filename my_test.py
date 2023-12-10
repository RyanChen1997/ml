# 分类水果

import os
from PIL import Image
import numpy as np
import sys

sys.path.append(os.curdir + "/【源代码】深度学习入门：基于Python的理论与实现")
from common.layers import *
from common.trainer import Trainer
from common.gradient import numerical_gradient
import pickle
from collections import OrderedDict

img_file_path_train = "/Users/ryan/Documents/fruit/archive/train_zip"
img_file_path_test = "/Users/ryan/Documents/fruit/archive/test_zip"
image_shape = (100, 100)
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
        # if not _valid_image(data):
        #     continue
        data_list.append(data)
        label_list.append(_read_label(path))
    data_list = np.stack(data_list, axis=0)
    label_list = np.stack(label_list, axis=0)

    return data_list, label_list


def _read_img(file_name):
    # 读取图片
    image = Image.open(file_name)
    # image = image.resize(image_shape, Image.Resampling.BILINEAR)
    image = image.resize(image_shape)
    image = image.convert("RGB")

    # 将图片转换为numpy数组
    image_array = np.array(image)
    image_array = image_array.transpose(2, 0, 1)
    # print(image_array.shape)

    # 将三维数组展平为一维数组
    # flat_image_array = image_array.flatten()
    return image_array


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
    T = np.zeros((X.size, 4))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def _valid_image(data):
    return data.shape[0] / image_shape[0] / image_shape[1] == 3.0


class TwoLayerNet:
    def __init__(self) -> None:
        pass


class SimpleConvNet:
    """简单的ConvNet

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        # 初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """求损失函数
        参数x是输入数据、t是教师标签
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=10):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads["W" + str(idx)] = numerical_gradient(
                loss_w, self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = numerical_gradient(
                loss_w, self.params["b" + str(idx)]
            )

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i + 1)]
            self.layers[key].b = self.params["b" + str(i + 1)]


def train():
    (x_train, t_train), (x_test, t_test) = load_data(one_hot_label=True, normalize=True)

    input_dim = x_train[0].shape
    conv_param = {
        "filter_num": 30,
        "filter_size": 15,
        "pad": 0,
        "stride": 1,
    }
    hidden_size = 100
    output_size = 4
    network = SimpleConvNet(
        input_dim=input_dim,
        conv_param=conv_param,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    max_epochs = 20
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=max_epochs,
        mini_batch_size=5,
        optimizer="Adam",
        optimizer_param={"lr": 0.001},
        evaluate_sample_num_per_epoch=20,
    )
    trainer.train()

    network.save_params("params.pkl")
    print("Save model.")


def result_to_label(result):
    idx = np.argmax(result)
    if idx == 0:
        return "apple"
    if idx == 1:
        return "banana"
    if idx == 2:
        return "orange"
    if idx == 3:
        return "mixed"


def predict():
    # 选一张图片
    image_path = "./mixed.jpeg"

    # 初始化network
    network = SimpleConvNet()
    network.load_params("params.pkl")

    # 准备数据
    x = _read_img(image_path)
    x = x.reshape(1, 3, 100, 100)

    y = network.predict(x)

    print(result_to_label(y))


if __name__ == "__main__":
    # train()
    predict()
