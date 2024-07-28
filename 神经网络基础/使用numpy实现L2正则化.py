# -*- encoding:utf-8 -*-
# editor:踩着上帝的小丑
# time:2024/7/16
# file:使用numpy实现L2正则化.py
from sklearn import datasets
from sklearn.linear_model import Ridge, LinearRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_load = datasets.load_boston()
x_train, x_test, y_train, y_test = train_test_split(data_load.data, data_load.target, test_size=0.3)
# 线性模型（不做处理）
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print('不做处理的线性模型', score)
# 标准化处理
standard = StandardScaler()
x_train_norm = standard.fit_transform(x_train)
x_test_norm = standard.transform(x_test)
model.fit(x_train_norm, y_train)
y_pred = model.predict(x_test_norm)
score = model.score(x_test_norm, y_test)
print('标准化的线性模型', score)

# L2正则化处理
ridge = Ridge()
ridge.fit(x_train_norm, y_train)
y_pred = ridge.predict(x_test_norm)
score = ridge.score(x_test_norm, y_test)
print('L2正则化的线性模型', score)
# L1正则化处理
lasso = Lasso()
lasso.fit(x_train_norm, y_train)
y_pred = lasso.predict(x_test_norm)
score = lasso.score(x_test_norm, y_test)
print('L1正则化的线性模型', score)