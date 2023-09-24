# 加载数据集
import pandas as pd
import numpy as np


def load_data(file_path):
    features = []  # 存放样本特征
    labels = []  # 存放样本标签
    data = pd.read_csv(file_path)  # pd.DataFrame格式
    data_list = data.values.tolist()
    for item in data_list:
        labels.append(item[0])
        features.append(item[1:])
    return features, labels


def calculate_dist(x1, x2):
    return np.square(np.sum(np.square(x1 - x2)))


def get_closest(x, features, labels, k):
    dist_list = [0] * len(labels)
    for i, feat in enumerate(features):
        dist = calculate_dist(np.array(x), np.array([feat]))
        dist_list[i] = dist
    top_index = np.argsort(np.array(dist_list))[:k]
    labels_array = np.array(labels)
    candidates = labels_array[top_index]
    res = np.bincount(candidates).argmax()
    return res


def get_accuracy(train_features, train_labels, test_features, test_labels, k):
    errors = 0
    for i in range(len(test_features)):
        x = test_features[i]
        y = get_closest(x, train_features, train_labels, k)
        if y != test_labels[i]: errors += 1
    return 1 - (errors / len(test_labels))


if __name__ == '__main__':
    print("开始")
    f, l = load_data(r'../data/mnist_train.csv')
    f1, l1 = load_data(r'../data/mnist_test.csv')
    print('accuracy' + get_accuracy(f, l, f1, l1, 5))
