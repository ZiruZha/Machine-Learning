# -*- coding: utf-8 -*-
# @Time : 2021/8/1 0:06
# @Author : ZiruZha
# @File : 20newsgroups-NaiveBayesClassifier.py.py
# @Software : PyCharm


"""
Naive Bayes classifiers
In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features (see Bayes classifier). They are among the simplest Bayesian network models,[1] but coupled with kernel density estimation, they can achieve higher accuracy levels.
"""
"""
朴素贝叶斯算法
优点：
    分类效率稳定
    对缺失数据不敏感，算法简单，常用于文本分类
    分类准确度高，速度快
缺点：
    算法假设样本属性独立，当特征属性有关联时，效果不好
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def naive_bayesian_news():
    """
    用朴素贝叶斯算法分类新闻
    :return:
    """
    # 1. 获取数据
    news = fetch_20newsgroups(subset="all")
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # 3. 特征工程：文本特征抽取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4. 朴素贝叶斯算法预估器
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)
    # 5. 模型评估
    # 方法一：直接比对真实值和预估值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值：", y_predict == y_test)
    # 方法二：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：", score)
    return None


if __name__ == "__main__":
    naive_bayesian_news()
    pass
