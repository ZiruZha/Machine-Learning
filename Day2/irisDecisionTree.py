# -*- coding: utf-8 -*-
# @Time : 2021/7/30 15:55
# @Author : ZiruZha
# @File : test3.py
# @Software: PyCharm
"""
鸢尾花决策树
iris classification using Decision Tree
"""
"""
决策树：
有点：
    可视化，可解释能力强
缺点：
    容易产生过拟合
"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# 决策树可视化
from sklearn.tree import export_graphviz


def iris_dt():
    """
    决策树分类鸢尾花
    :return:
    """
    # 1. 获取数据集
    iris = load_iris()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 3. 决策树预估器分类
    # sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
    # criterion默认是'gini'，也可以选择信息增益的熵'entropy'
    # max_depth树的深度
    # random_state随机数种子
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)
    # 4. 模型评估
    # 方法一：直接比对真实值和预估值
    y_predict = estimator.predict(x_test)
    print("y_predoct:\n", y_predict)
    print("直接比对真实值和预测值：", y_predict == y_test)
    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)

    # 可视化决策树
    # 用网页打开生成的.dot文件
    # 网页链接
    # webgraphviz.com
    export_graphviz(estimator, out_file="iris_decision_tree.dot", feature_names=iris.feature_names)

    return None


if __name__ == "__main__":
    iris_dt()
    pass
