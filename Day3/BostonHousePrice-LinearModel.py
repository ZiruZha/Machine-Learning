# -*- coding: utf-8 -*-
# @Time : 2021/8/2 10:18
# @Author : ZiruZha
# @File : test1.py
# @Software: PyCharm


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# 保存模型
import joblib


def linear1():
    """
    正规方程法
    :return:
    """
    # 1. 获取数据
    boston = load_boston()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3. 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4. 预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    # 5. 得出模型
    print("正规方程-权重系数为：\n", estimator.coef_)
    print("正规方程-截距为：\n", estimator.intercept_)
    # 6. 模型评估
    y_predict = estimator.predict(x_test)
    print("正规方程-预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差：\n", error)
    return None


def linear2():
    """
    梯度下降法
    :return:
    """
    # 1. 获取数据
    boston = load_boston()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3. 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # # 4. 预估器
    # estimator = SGDRegressor()
    # estimator.fit(x_train, y_train)
    # # 4.5 保存模型
    # joblib.dump(estimator, "my_SGDRegressor.pkl")
    # 4.5 加载模型
    estimator = joblib.load("my_SGDRegressor.pkl")
    # 5. 得出模型
    print("梯度下降-权重系数为：\n", estimator.coef_)
    print("梯度下降-截距为：\n", estimator.intercept_)
    # 6. 模型评估
    y_predict = estimator.predict(x_test)
    print("梯度下降-预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降-均方误差：\n", error)
    return None


def linear3():
    """
    岭回归
    ridge
    :return:
    """
    # 1. 获取数据
    boston = load_boston()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    # 3. 标准化
    transfer = Ridge()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4. 预估器
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)
    # 5. 得出模型
    print("岭回归-权重系数为：\n", estimator.coef_)
    print("岭回归-截距为：\n", estimator.intercept_)
    # 6. 模型评估
    y_predict = estimator.predict(x_test)
    print("岭回归-预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归-均方误差：\n", error)
    return None


if __name__ == "__main__":
    # 正规方程法
    linear1()
    # 梯度下降法
    linear2()
    # 岭回归
    linear3()
    pass
