from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
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


# 加载数据
f, l = load_data(r'../data/mnist_train.csv')
f1, l1 = load_data(r'../data/mnist_test.csv')

# 创建感知机模型
perceptron = Perceptron(tol=1e-3, random_state=0)

# 在训练集上训练模型
perceptron.fit(f, l)

# 在测试集上进行预测
y_pred = perceptron.predict(f1)

# 计算准确率
accuracy = accuracy_score(l1, y_pred)
print("Accuracy:", accuracy)
