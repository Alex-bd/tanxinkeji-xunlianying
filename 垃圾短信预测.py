#
# 把文本转为数字 预处理  （给定单词求频率）
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("文件.txt", delimiter='\t', header=None)
y, X_train = df[0], df[1]

vectorizer = TfidfVectorizer()  # 向量化
x = vectorizer.fit_transform(X_train)  # 利用向量化的变量把文本转为数字特征

lr = linear_model.LogisticRegression()  # 生成一个逻辑回归模型
lr.fit(x, y)  # 对数字特征和y标签进行训练

# 生成一条测试数据
testX = vectorizer.transform(['URGENT! your mobile No.1234 was awarded a '])

predictions = lr.predict(testX)  # 预测
print(predictions)


