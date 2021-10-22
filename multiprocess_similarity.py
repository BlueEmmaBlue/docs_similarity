import math
import time
import numpy as np
import re
from string import punctuation as punctuation_en
from zhon.hanzi import punctuation as punctuation_zh
from collections import Counter
from multiprocessing import Pool, cpu_count, Manager


def tf(tmp_word, tmp_dict):
    return tmp_dict[tmp_word] / sum(tmp_dict.values())  # tf = 一个word在一篇文章中出现的次数 / 文章中所有词语数之和


def how_many_included(tmp_word, tmp_dicts):  # 计算一个word在多少篇文章中出现过
    tmp_sum = 0
    for tmp_dict in tmp_dicts:
        if tmp_word in tmp_dict:
            tmp_sum = tmp_sum + 1
    return tmp_sum


def idf(tmp_word, tmp_dicts):  # idf = log(文章总数 / (包含这个word的文章数 + 1))
    return math.log((len(tmp_dicts) / (how_many_included(tmp_word, tmp_dicts)) + 1), 10)


def tf_idf(tmp_word, tmp_dict, tmp_dicts):  # tf * idf
    return tf(tmp_word, tmp_dict) * idf(tmp_word, tmp_dicts)


def cal_similarity(x, y):  # 向量x和y的余弦相似度计算
    xx = np.sum(x ** 2) ** 0.5  # 计算向量x的长度
    x = x / xx  # 用 x/x的长度
    yy = np.sum(y ** 2) ** 0.5  # 计算向量y的长度
    y = y / yy  # 用 y/y的长度
    similarity = np.dot(x, y)  # 再对向量进行点乘
    return similarity


def cal_tf_idf(tmp_dict, tmp_dicts):
    scores = {}
    for tmp_word in tmp_dict:  # 遍历每篇文章中的word
        scores[tmp_word] = tf_idf(tmp_word, tmp_dict, tmp_dicts)  # 计算每个word的tf-idf指数
    sorted_scores = sorted(scores.items(), key=lambda s: s[1], reverse=True)  # 对每篇文章的word的tf-idf指数倒序排列
    return sorted_scores[0:10]  # 取前十个作为这篇文章的关键词


if __name__ == '__main__':
    cpus = cpu_count()

    file = []
    f = open("/Users/mac/Documents/docs_similarity/test.txt",
             "r", encoding="gbk")
    file = f.read()
    f.close()

    # 预处理
    start = time.time()  # 计算开始时间
    docs = file.split("\n\n")  # 通过两个换行符区分不同文章
    res = '[a-zA-Z0-9]'  # 为了去除文章前方的id编号，对英文字母和数字做正则匹配
    docs_words = []  # 所有文章的所有词语合集
    dicts = Manager().list()  # 所有文章的词语计数后合集
    for doc in docs:  # 对每篇文章
        doc_words = []  # 初始化每篇文章的词语合集
        lines = doc.split("\n")  # 通过换行符区分不同的行
        if len(lines) == 0:
            continue
        line_1 = lines[0]
        id = line_1[0:15]  # 每篇文章的id
        for line in lines:  # 遍历每一行
            line = re.sub("[{}]+".format(punctuation_en), "", line)  # 删除所有英文符号
            line = re.sub("[{}]+".format(punctuation_zh), "", line)  # 删除所有中文富豪
            line = re.sub(res, "", line)  # 删除所有英文字母和数字
            words = line.split(' ')  # 按照空格分割单词
            words = list(filter(None, words))  # 去掉所有空字符串
            for word in words:  # 把一篇文章的单词集合放入doc_words中
                doc_words.append(word)
        docs_words.append(doc_words)  # 把所有文章的单词集合放入docs_words中
        dict = Counter(doc_words)  # 使用Counter把对一篇单词进行计数，并放入dict中
        dicts.append(dict)  # 把所有文章的单词计数之后的结果放入dicts中
    end = time.time()  # 计算总用时
    print("对每篇文章分词共用时：", end - start)

    pool = Pool(cpus)
    tf_idf_list = Manager().list()

    # 计算每篇文章词语的tf*idf并降序排列
    start = time.time()  # 计算开始时间
    for dict in dicts:  # 对dicts循环
        tmp = pool.apply_async(cal_tf_idf, args=(dict, dicts))
        tf_idf_list.append(tmp.get())
    pool.close()
    pool.join()
    end = time.time()
    print("计算tf-idf用时：", end - start)  # 计算总用时

    # 基于关键词的one-hot向量构建每篇文章对应的向量
    start = time.time()  # 计算开始时间
    keywords = set()  # 创建关键词集合
    for tf_idf_doc in tf_idf_list:  # 对于每篇文章中关键词list
        for word in tf_idf_doc:  # 对其中的每个关键词
            keywords.add(word[0])  # 在关键词集合中加入该关键词，利用集合的特性去重
    keywords = list(keywords)  # 将集合转换为list
    onehot_vectors = np.zeros((len(docs_words), len(keywords)), dtype=np.int64)  # 初始化所有文章的onehot向量
    for i, doc_words in enumerate(docs_words):  # 遍历每篇文章
        for j, word in enumerate(doc_words):  # 遍历文章中每个word
            if word in keywords:  # 如果这个word在关键词集合中出现过，则将这个该关键词向量下标对应的值赋为1
                onehot_vectors[i][keywords.index(word)] = 1
    end = time.time()  # 计算总用时
    print("构建每个文章向量用时：", end - start)

    # 计算两两间相似度
    pool = Pool(cpus)
    start = time.time()  # 计算开始时间
    len_docs = len(docs_words)  # 计算文章总数
    similarity_metrix = np.zeros((len_docs, len_docs))  # 初始化文章间两两相似矩阵
    for i, onehot_vector_i in enumerate(onehot_vectors):
        for j, onehot_vector_j in enumerate(onehot_vectors):  # 计算文章i和文章j之间的余弦相似度
            tmp = pool.apply_async(cal_similarity, args=(onehot_vector_i, onehot_vector_j))
            similarity_metrix[i, j] = tmp.get();
    pool.close()
    pool.join()
    end = time.time()  # 计算总用时
    print(similarity_metrix)
    print("计算两两相似度用时：", end - start)
