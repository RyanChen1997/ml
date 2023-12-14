import os
from PIL import Image
import numpy as np

img_file_path_train = "/home/cyx/document/archive/train_zip"
img_file_path_test = "/home/cyx/document/archive/test_zip"
image_shape = (100, 100)


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

def read_img(file_name):
    return _read_img(file_name)
