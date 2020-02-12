import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score  # 计算交叉验证的
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder  # 把非数值的数据转为数值
from keras.models import model_from_json

# reproducibility
seed = 13
np.random.seed(seed)  # 设置随机数种子

# load data
df = pd.read_csv('iris.csv')
X = df.values[:, 0:4].astype(float)
Y = df.values[:, 4]

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)  # 把字符串变为数值
# Y_onehot = np_utils.to_categorical(Y_encoded)
Y_onehot = np_utils.to_categorical(Y)

# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Dense(7, input_dim=4, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=2, verbose=1)

# 评估
# 10份中9份训练，一份测试
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
print('Accuray of cross validation,mean %.2f,std %.2f' % (result.mean(), result.std()))

# 保存模型
estimator.fit(X, Y_onehot)
model_json = estimator.model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
estimator.model.save_weights('model.h5')
print('saved model to disk')

# 导入模型 并且使用它做预测
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model from disk')

predicted = loaded_model.predict(X)
print('perdicted probability:' + str(predicted), end='')

predicted_label = loaded_model.predict_classes(X)
print('predicted label:' + str(predicted_label))