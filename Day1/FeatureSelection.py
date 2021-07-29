# -*- coding: utf-8 -*-
# @Time : 2021/7/29 9:06
# @Author : ZiruZha
# @File : FeatureSelection.py
# @Software: PyCharm

"""
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
"""
"""
两种降维方法：
1. 特征选择
    1.1 filter
        方差选择法
        相关系数
    1.2 embedded
        决策树
        正则化
        深度学习
2. 主成分分析
"""


import pandas as pd
from sklearn.feature_selection import VarianceThreshold
# 计算皮尔森相关系数
from scipy.stats import pearsonr
# 降维
from sklearn.decomposition import PCA

"""
Feature selection
The classes in the sklearn.feature_selection module can be used for feature selection/dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.
"""
def variance_demo():
    """
    低方差特征过滤
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 2:-2]
    print("data:\n", data)
    # 2. 实例化一个转换器类
    # 可以设置阈值参数
    transfer = VarianceThreshold(threshold=5)
    # 3. 调用fit_transform
    # 若data中包含字符或字符串，则报错
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)
    # 4. 计算某两个变量之间的相关系数
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数：", r1)
    r2 = pearsonr(data["revenue"], data["total_expense"])
    print("相关系数：", r2)
    # 对于相关性较强的两组数据，可以采取如下措施
    # 1. 选取其中一个
    # 2. 加权求和
    # 3. 主成分分析
    return None

"""
The sklearn.decomposition module includes matrix decomposition algorithms, including among others PCA, NMF or ICA. Most of the algorithms of this module can be regarded as dimensionality reduction techniques.
"""
def pca_demo():
    """
    pca降维
    :return:
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # 参数n_components可以为整数或分数
    # 分数：保留该分数的信息
    # 整数：减少到该整数的特征
    transfer = PCA(n_components=2)
    data_new = transfer.fit_transform(data)
    print("data_new:", data_new)
    return None



if __name__ == "__main__":
    # 低方差特征过滤
    # variance_demo()
    # pca降维
    pca_demo()
