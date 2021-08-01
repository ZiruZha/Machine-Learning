# -*- coding: utf-8 -*-
# @Time : 2021/7/31 16:55
# @Author : ZiruZha
# @File : facebook-v-predicting-check-ins.py
# @Software : PyCharm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def facebook():
    """
    facebook-v-predicting-check-ins
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("train.csv")
    # 2. 基本数据处理
    # 2.1 缩小数据范围(减少计算时间，调试用)
    # data = data.query("x < 2.5 & x > 2 & y < 1.5 & y > 1.0")
    # 2.2 处理时间特征
    time_value = pd.to_datetime(data["time"], unit="s")
    date = pd.DatetimeIndex(time_value)
    data.loc[:, "day"] = date.day
    data.loc[:, "weekday"] = date.weekday
    data.loc[:, "hour"] = date.hour
    # print(data)
    # 2.3 过滤签到次数少的地点
    place_count = data.groupby("place_id").count()["row_id"]
    place_count_filtered = place_count[place_count > 3]
    data_final = data[data["place_id"].isin(place_count_filtered.index.values)]
    # print(data_final)
    # 2.4 筛选特征值和目标值
    x = data_final[["x", "y", "accuracy", "day", "weekday", "hour"]]
    y = data_final["place_id"]
    # print(x, "\n")
    # print(y)

    # 2.5 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 3. 特征工程（标准化）
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. KNN算法预估器
    estimator = KNeighborsClassifier()

    # 4.5 加入网格搜索与交叉验证（新增）
    # 参数准备
    param_grid = {"n_neighbors": [3, 5, 7]}
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
    facebook()
    pass
