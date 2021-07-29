# -*- coding: utf-8 -*-
# @Time : 2021/7/28 21:28
# @Author : ZiruZha
# @File : Preprocessing.py
# @Software: PyCharm


"""
预处理
Preprocessing data
The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.

In general, learning algorithms benefit from standardization of the data set. If some outliers are present in the set, robust scalers or transformers are more appropriate. The behaviors of the different scalers, transformers, and normalizers on a dataset containing marginal outliers is highlighted in Compare the effect of different scalers on data with outliers.
"""
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
}
"""


# 调用归一化
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 读取数据库函数
import pandas as pd


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1. 获取数据
    # 从当前目录下读取数据
    data = pd.read_csv("data.csv")
    # 只对特征进行归一化，不对target归一化，即取前三列
    data = data.iloc[:, :3]
    print("data:\n", data)
    # 2. 实例化一个转换器类
    # MinMaxScaler()默认归一到[0, 1]，有参数feature_range调节
    transfer = MinMaxScaler(feature_range=[2, 3])
    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


# 归一化数据易受数据种个别极值的影响，标准化可减小该影响
def standard_demo():
    """
    标准化
    :return:
    """
    # 1. 获取数据
    # 从当前目录下读取数据
    data = pd.read_csv("data.csv")
    # 只对特征进行归一化，不对target归一化，即取前三列
    data = data.iloc[:, :3]
    print("data:\n", data)
    # 2. 实例化一个转换器类
    transfer = StandardScaler()
    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None


if __name__ == "__main__":
    # 归一化
    # minmax_demo()
    # 标准化
    standard_demo()
