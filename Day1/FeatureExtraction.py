# -*- coding: utf-8 -*-
# @Time : 2021/7/28 11:15
# @Author : ZiruZha
# @File : FeatureExtraction.py
# @Software: PyCharm


"""
特征提取
Feature extraction
The sklearn.feature_extraction module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image.

Note Feature extraction is very different from Feature selection: the former consists in transforming arbitrary data, such as text or images, into numerical features usable for machine learning. The latter is a machine learning technique applied on these features.
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

# 导入数据库
from sklearn.datasets import load_iris
# 导入数据库分类函数
from sklearn.model_selection import train_test_split
# 导入字典特征提取函数
from sklearn.feature_extraction import DictVectorizer
# 导入文本特征提取函数
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 分词处理
import jieba


def datasets_demo():
    # sklearn数据集使用
    # 获取数据集
    # 小规模数据用load，大规模数据用fetch
    iris = load_iris()
    """
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris["DESCR"])
    print("查看特征值名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)
    """

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'name': 'a', 'age': 1}, {'name': 'b', 'age': 2}, {'name': 'c', 'age': 3}]
    # 1. 实例化一个转换器
    # DictVectorizer()函数默认返回sparse矩阵（稀疏矩阵）——给出矩阵中非零值的位置与值
    # 当类别较多时，sparse矩阵可以节省内存
    # 用one-hot编码表示类别
    # One-Hot编码，又称一位有效编码。其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效。
    transform = DictVectorizer(sparse=False)
    # 2. 调用fit_transform()方法，传入字典参数
    data_new = transform.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名字\n", transform.get_feature_names())
    return None


def count_demo():
    """
    英文文本特征抽取：CountVectorizer()
    统计每个单词特征值出现的次数
    :return:
    """
    data = ["To be or not to be", "I have a pen"]
    # 1. 实例化一个转换器类
    # CountVectorizer()不统计单个字母单词，如I a
    # CountVectorizer()只能返回sparse矩阵
    # CountVectorizer()无sparse参数，
    # 停用词stop_words参数，列表，不统计列表包含的单词
    transform = CountVectorizer(stop_words=["be", "to"])
    # 2. 调用fit_transform()方法
    data_new = transform.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字\n", transform.get_feature_names())
    return None


def count_chinese_demo():
    """
    中文文本特征抽取：CountVectorizer()
    统计每个短语特征值出现的次数
    需要进行分字处理
    :return:
    """
    data = ["一二三四五", "上山打老虎"]
    # 1. 实例化一个转换器类
    # CountVectorizer()只能返回sparse矩阵
    # CountVectorizer()无sparse参数，
    transform = CountVectorizer()
    # 2. 调用fit_transform()方法
    data_new = transform.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字\n", transform.get_feature_names())
    return None


def cut_word(text):
    """
    进行中文分词：”我爱北京天安门“-->”我 爱 北京 天安门“
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    """
    中文文本特征提取，自动分词
    :return:
    """
    data = [
        "鲁镇的酒店的格局，是和别处不同的：都是当街一个曲尺形的大柜台，柜里面预备着热水，可以随时温酒。做工的人，傍午傍晚散了工，每每花四文铜钱，买一碗酒，——这是二十多年前的事，现在每碗要涨到十文，——靠柜外站着，热热的喝了休息；倘肯多花一文，便可以买一碟盐煮笋，或者茴香豆，做下酒物了，如果出到十几文，那就能买一样荤菜，但这些顾客，多是短衣帮，大抵没有这样阔绰。只有穿长衫的，才踱进店面隔壁的房子里，要酒要菜，慢慢地坐喝。",
        "我从十二岁起，便在镇口的咸亨酒店里当伙计，掌柜说，我样子太傻，怕侍候不了长衫主顾，就在外面做点事罢。外面的短衣主顾，虽然容易说话，但唠唠叨叨缠夹不清的也很不少。他们往往要亲眼看着黄酒从坛子里舀出，看过壶子底里有水没有，又亲看将壶子放在热水里，然后放心：在这严重监督下，羼水也很为难。所以过了几天，掌柜又说我干不了这事。幸亏荐头的情面大，辞退不得，便改为专管温酒的一种无聊职务了。",
        "我从此便整天的站在柜台里，专管我的职务。虽然没有什么失职，但总觉得有些单调，有些无聊。掌柜是一副凶脸孔，主顾也没有好声气，教人活泼不得；只有孔乙己到店，才可以笑几声，所以至今还记得。"]
    # 0. 将中文文本进行分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1. 实例化一个转换器类
    # CountVectorizer()只能返回sparse矩阵
    # CountVectorizer()无sparse参数，
    transform = CountVectorizer()
    # 2. 调用fit_transform()方法
    data_final = transform.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字\n", transform.get_feature_names())
    return None


def tfidf_demo():
    """
    用tfidf函数进行文本特征提取
    tfidf: TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.
    :return:
    """
    data = [
        "鲁镇的酒店的格局，是和别处不同的：都是当街一个曲尺形的大柜台，柜里面预备着热水，可以随时温酒。做工的人，傍午傍晚散了工，每每花四文铜钱，买一碗酒，——这是二十多年前的事，现在每碗要涨到十文，——靠柜外站着，热热的喝了休息；倘肯多花一文，便可以买一碟盐煮笋，或者茴香豆，做下酒物了，如果出到十几文，那就能买一样荤菜，但这些顾客，多是短衣帮，大抵没有这样阔绰。只有穿长衫的，才踱进店面隔壁的房子里，要酒要菜，慢慢地坐喝。",
        "我从十二岁起，便在镇口的咸亨酒店里当伙计，掌柜说，我样子太傻，怕侍候不了长衫主顾，就在外面做点事罢。外面的短衣主顾，虽然容易说话，但唠唠叨叨缠夹不清的也很不少。他们往往要亲眼看着黄酒从坛子里舀出，看过壶子底里有水没有，又亲看将壶子放在热水里，然后放心：在这严重监督下，羼水也很为难。所以过了几天，掌柜又说我干不了这事。幸亏荐头的情面大，辞退不得，便改为专管温酒的一种无聊职务了。",
        "我从此便整天的站在柜台里，专管我的职务。虽然没有什么失职，但总觉得有些单调，有些无聊。掌柜是一副凶脸孔，主顾也没有好声气，教人活泼不得；只有孔乙己到店，才可以笑几声，所以至今还记得。"]
    # 0. 将中文文本进行分词
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1. 实例化一个转换器类
    transform = TfidfVectorizer()
    # 2. 调用fit_transform()方法
    data_final = transform.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字\n", transform.get_feature_names())
    return None


if __name__ == "__main__":
    # sklearn数据集使用
    # datasets_demo()
    # 字典特征抽取
    # dict_demo()
    # 英文本特征抽取：CountVectorizer()
    # count_demo()
    # 中文本特征抽取：CountVectorizer()
    # count_chinese_demo()
    # count_chinese_demo2()
    # 中文分词
    # print(cut_word("我爱北京天安门"))
    # tfidf 文本特征提取
    tfidf_demo()
