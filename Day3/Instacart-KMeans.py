# -*- coding: utf-8 -*-
# @Time : 2021/8/2 15:47
# @Author : ZiruZha
# @File : test3.py
# @Software: PyCharm


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def instacart():
    """
    K-means
    特点：采用迭代算法，直观易懂，实用
    缺点：易于陷入局部最优
    :return:
    """
    # 1. 获取数据
    order_products__prior = pd.read_csv("./instacart_data/order_products__prior.csv")
    aisles = pd.read_csv("./instacart_data/aisles.csv")
    orders = pd.read_csv("./instacart_data/orders.csv")
    products = pd.read_csv("./instacart_data/products.csv")
    # 2. 合并表
    # 2.1 合并aisles和products
    table1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
    # 2.2 合并table1和order_products__prior
    table2 = pd.merge(table1, order_products__prior, on=["product_id", "product_id"])
    # 2.3 合并table2和orders
    table3 = pd.merge(table2, orders, on=["order_id", "order_id"])

    # 3. 找到user_id和aisle之间的关系
    table = pd.crosstab(table3["user_id"], table3["aisle"])
    # 数据过大
    data = table[:10000]
    # 4. PCA降维
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    # print(data_new.shape)
    # 5. 预估器
    estimator = KMeans(n_clusters=3)
    estimator.fit(data_new)
    y_predict = estimator.predict(data_new)
    # print(y_predict)
    # 6. 模型评估——轮廓系数
    score = silhouette_score(data_new, y_predict)
    print(score)
    return None


if __name__ == "__main__":
    instacart()
    pass
