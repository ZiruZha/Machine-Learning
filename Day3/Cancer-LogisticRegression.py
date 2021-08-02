# -*- coding: utf-8 -*-
# @Time : 2021/8/2 14:19
# @Author : ZiruZha
# @File : test2.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import numpy as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# 查看精确率、召回率、F1-score
from sklearn.metrics import classification_report
# roc_auc
from sklearn.metrics import roc_auc_score


def cancer_demo():
    """
    逻辑回归分类癌症
    :return:
    """
    # 1. 读取数据
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("breast-cancer-wisconsin.data", names=column_name)
    # print(data)

    # 2. 缺失值处理
    # 2.1 替换-->np.nan
    data = data.replace(to_replace="?", value=np.NaN)
    # 2.2 删除缺失样本
    data.dropna(inplace=True)
    # print(data.isnull().any())

    # 3. 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]
    # 4. 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # 5. 特征工程
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 6. 预估器
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 逻辑回归的模型参数：回归系数和截距
    print("逻辑回归的回归系数：", estimator.coef_)
    print("逻辑回归的截距：", estimator.intercept_)
    # 7. 模型评估
    # 7.1 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:", y_predict)
    print("y_predict == y_test?:", y_predict == y_test)
    # 7.2 计算准确率
    score = estimator.score(x_test, y_test)
    print("score:", score)
    # 7.3 精确率、召回率、F1-score
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["benign-良性", "malignant-恶性"])
    print(report)
    # 7.4 ROC-AUC
    # 对于不均衡样本，需使用这一评估方法
    # 接受者操作特性曲线（receiver operating characteristic curve，简称ROC曲线），又称为感受性曲线（sensitivity curve）。得此名的原因在于曲线上各点反映着相同的感受性，它们都是对同一信号刺激的反应，只不过是在几种不同的判定标准下所得的结果而已。接受者操作特性曲线就是以虚惊概率为横轴，击中概率为纵轴所组成的坐标图，和被试在特定刺激条件下由于采用不同的判断标准得出的不同结果画出的曲线
    # AUC（Area Under Curve）被定义为ROC曲线下的面积。
    # AUC只能用来评价二分类
    # AUC适合评价样本不平衡中的分类器性能

    # y_true：每个样本的真实类别，必须为0（反例），1（正例）标记
    # 将y_test转化为0 1
    y_true = np.where(y_test > 3, 1, 0)
    roc_auc_score(y_true, y_predict)
    return None


if __name__ == "__main__":
    cancer_demo()
    pass
