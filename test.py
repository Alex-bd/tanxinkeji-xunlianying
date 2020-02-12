# 逻辑回归 二分类问题

from sklearn import linear_model
X = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]
y = [0, 1, 1, 1, 0, 0]
lr = linear_model.LogisticRegression()
# 训练
lr.fit(X, y)

testX = [[28, 8]]
# 预测
label = lr.predict(testX)
print("predicted Label =", label)
prob = lr.predict_log_proba(testX)
print("概率：", prob)

#
theta_0 = lr.intercept_
theta_1 = lr.coef_[0][0]
theta_2 = lr.coef_[0][1]

print("theta_0=",theta_0)
print("theta_1=",theta_1)
print("theta_2=",theta_2)

testX = [[28, 8]]
ratio = prob[1] / prob[0]