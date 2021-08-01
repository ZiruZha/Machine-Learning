# -*- coding: utf-8 -*-
# @Time : 2021/7/29 16:30
# @Author : ZiruZha
# @File : test.py
# @Software: PyCharm


# 调用数据集
from sklearn.datasets import load_iris
# 数据集划分
from sklearn.model_selection import train_test_split
# 标准化
from sklearn.preprocessing import StandardScaler
# KNN
from sklearn.neighbors import KNeighborsClassifier
# gird search and cross validation
from sklearn.model_selection import GridSearchCV

"""
K邻近算法（K-NearestNeighbor）
优点：
    简单、易于理解、易于实现、无需训练
缺点：
    必须指定k值，k值选取影响分类精度
    懒惰算法，对测试样本分类时的计算量大，内存开销大
"""


def knn_iris():
    """
    Classification of iris using KNN
    :return:
    """
    # 1. 获取数据

    iris = load_iris()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # print("iris:", iris)
    # print("x_train:", x_train)
    # print("x_test:", x_test)
    # print("y_train:", y_train)
    # print("y_test:", y_test)

    # 3. 特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. KNN算法预估器
    # 如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别，则该样本也属于这个类别。
    # k = 参数n_neighbors，默认为5，
    # k值过小，易受到异常值的影响
    # k值过大，样本不均衡的影响
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    # 方法一：直接比对真实值和预估值
    y_predict = estimator.predict(x_test)
    print("y_predoct:\n", y_predict)
    print("直接比对真实值和预测值：", y_predict == y_test)
    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)


    return None


"""
网格搜索：gird search
自动选择KNN算法中最优参数K
交叉验证：cross validation
让评估模型结果更加准确
"""


def knn_iris_gscv():
    """
    模型调优，添加网格搜索和交叉验证
    :return:
    """
    # 1. 获取数据

    iris = load_iris()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=1)

    # 3. 特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. KNN算法预估器
    # 如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别，则该样本也属于这个类别。
    # k = 参数n_neighbors，默认为5，
    # k值过小，易受到异常值的影响
    # k值过大，样本不均衡的影响
    estimator = KNeighborsClassifier()

    # 4.5 加入网格搜索与交叉验证（新增）
    # sklearn.model_selection.GridSearchCV(estimator=, param_grid=, cv=)
    # estimator估计器对象
    # param_grid估计器参数(dict){"n_neighbors": [1, 3, 5]}
    # cv指定几折交叉验证
    # 结果分析：
    #     最佳参数：best_params_
    #     最佳结果：best_score_
    #     最佳估计器：best_estimator_
    #     交叉验证结果：cv_results_

    # 参数准备
    param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10)
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    # 方法一：直接比对真实值和预估值
    y_predict = estimator.predict(x_test)
    print("y_predoct:\n", y_predict)
    print("直接比对真实值和预测值：", y_predict == y_test)
    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)

    # 最佳参数：best_params_
    print("最佳参数\n", estimator.best_params_)
    # 最佳结果：best_score_
    print("最佳结果\n", estimator.best_score_)
    # 最佳估计器：best_estimator_
    print("最佳估计器\n", estimator.best_estimator_)
    # 交叉验证结果：cv_results_
    print("交叉验证结果\n", estimator.cv_results_)
    return None


if __name__ == "__main__":
    # KNN
    # knn_iris()
    # KNN with gird search and cross validation
    knn_iris_gscv()
    pass
