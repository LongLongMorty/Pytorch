import pandas as pd
import numpy as np

from tqdm import tqdm


def load_data(file_path):
    features = []  # 存放样本特征
    labels = []  # 存放样本标签
    data = pd.read_csv(file_path)  # pd.DataFrame格式
    data_list = data.values.tolist()
    for item in data_list:
        labels.append(item[0])
        features.append(item[1:])
    return features, labels


# 核心
def preception(features, labels, total_epoch=50, lr=0.001):
    # 参数初始化
    features_dim = len(features[0])
    w = np.zeros((1, features_dim))
    b = 0

    for epoch in tqdm(range(total_epoch)):
        for i in range(len(labels)):
            xi = np.array(features[i])
            yi = labels[i]
            if -1 * yi * (np.dot(w, xi) + b) > 0:
                w = w + lr * yi * xi
                b = b + lr * yi

    return w, b


def model_test(features, labels, w, b):
    errors = 0
    for i in range(len(labels)):
        xi = features[i]
        yi = labels[i]
        result = -1 * yi * (np.dot(w, xi) + b)
        if np.any(result < 0):
            errors += 1
    return 1 - (errors / len(labels))


if __name__ == '__main__':
    print("开始")
    f, l = load_data(r'../data/mnist_train.csv')
    f1, l1 = load_data(r'../data/mnist_test.csv')
    w, b = preception(f, l)
    print(model_test(f1, l1, w, b))
