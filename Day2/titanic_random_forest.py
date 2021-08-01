# -*- coding: utf-8 -*-
# @Time : 2021/8/1 10:44
# @Author : ZiruZha
# @File : test5.py
# @Software : PyCharm


"""
随机森林
处理高维特征样本不需要降维
能够评估各个特征在分类问题上的重要性
"""


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier


def titanic():
    """
    决策树
    :return:
    """
    # 1. 获取数据
    titanic_data = pd.read_csv("titanic_train.csv")
    # 1.5 筛选特征值和目标值
    x = titanic_data[["Pclass", "Age", "Sex"]]
    y = titanic_data["Survived"]
    # print(x)
    # 2. 数据处理
    # 2.1 缺失值处理
    x["Age"].fillna(x["Age"].mean(), inplace=True)
    # 2.2 转换成字典
    x = x.to_dict(orient="records")
    # print(x)
    # 2.3 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 2.4 字典特征抽取
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 3. 决策树预估器分类
    # sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
    # criterion默认是'gini'，也可以选择信息增益的熵'entropy'
    # max_depth树的深度
    # random_state随机数种子
    estimator = DecisionTreeClassifier(criterion='entropy', max_depth=8)
    estimator.fit(x_train, y_train)
    # 4. 模型评估
    # 方法一：直接比对真实值和预估值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：", y_predict == y_test)
    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)

    # 可视化决策树
    # 用网页打开生成的.dot文件
    # 网页链接
    # webgraphviz.com
    export_graphviz(estimator, out_file="titanic_decision_tree.dot", feature_names=transfer.get_feature_names())


    return None


def titanic_random_forest():
    """
    随机森林
    :return:
    """
    # 1. 获取数据
    titanic_data = pd.read_csv("titanic_train.csv")
    # 1.5 筛选特征值和目标值
    x = titanic_data[["Pclass", "Age", "Sex"]]
    y = titanic_data["Survived"]
    # print(x)
    # 2. 数据处理
    # 2.1 缺失值处理
    x["Age"].fillna(x["Age"].mean(), inplace=True)
    # 2.2 转换成字典
    x = x.to_dict(orient="records")
    # print(x)
    # 2.3 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
    # 2.4 字典特征抽取
    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 3. 随机森林算法预估器
    estimator = RandomForestClassifier()

    # 4 加入网格搜索与交叉验证
    # 参数准备
    param_grid = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3)
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    # 方法一：直接比对真实值和预估值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
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
    # titanic()
    titanic_random_forest()
    pass
