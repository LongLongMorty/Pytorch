from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# 加载数据集
import pandas as pd



def load_data(file_path):
    features = []  # 存放样本特征
    labels = []  # 存放样本标签
    data = pd.read_csv(file_path)  # pd.DataFrame格式
    data_list = data.values.tolist()
    for item in data_list:
        labels.append(item[0])
        features.append(item[1:])
    return features, labels


def get_closest(x, features, labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    res = knn.predict([x])
    return res[0]





if __name__ == '__main__':
    print("开始")
    f, l = load_data(r'../data/mnist_train.csv')
    f1, l1 = load_data(r'../data/mnist_test.csv')
    print(get_closest(f1[0], f, l, 5))
