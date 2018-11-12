#coding:utf-8

import numpy as np

def createDataset():
    '''
    create dataset
    :return:
    '''
    learning_dataset = {"宝贝当家": [45, 2, 9, "喜剧片"],
                        "美人鱼": [21, 17, 5, "喜剧片"],
                        "澳门风云3": [54, 9, 11, "喜剧片"],
                        "功夫熊猫3": [39, 0, 31, "喜剧片"],
                        "谍影重重": [5, 2, 57, "动作片"],
                        "叶问3": [3, 2, 65, "动作片"],
                        "伦敦陷落": [2, 3, 55, "动作片"],
                        "我的特工爷爷": [6, 4, 21, "动作片"],
                        "奔爱": [7, 46, 4, "爱情片"],
                        "夜孔雀": [9, 39, 8, "爱情片"],
                        "代理情人": [9, 38, 2, "爱情片"],
                        "新步步惊心": [8, 34, 17, "爱情片"]}
    return learning_dataset

def KNN(learning_dataset, dataPoint, k):
    '''
    knn算法，返回k个邻居的类别和得到测试数据的类别
    :param learning_dataset:
    :param dataPoint:
    :param k:
    :return:
    '''
    dislist  =[]
    for key, v in learning_dataset.items():
        d = np.linalg.norm(np.array(v[:3]) - np.array(dataPoint))
        dislist.append([key, round(d, 2)])
    print(dislist)
    #按距离大小进行递增排序
    dislist.sort(key=lambda dis: dis[1])
    #选取距离最小的K个样本
    dislist = dislist[:k]
    print(dislist)
    #确定前K个样本所在类别出现的频率，并输出出现频率最高的类别
    labels = {"喜剧片":0, "动作片":0, "爱情片":0}
    for s in dislist:
        label = learning_dataset[s[0]]
        labels[label[len(label) - 1]] += 1
    labels =sorted(labels.items(), key=lambda asd:asd[1], reverse=True)

    return labels, labels[0][0]

if __name__=='__main__':
    learining_dataset = createDataset()
    testData={"唐人街探案": [23, 3, 17, "？片"]}
    dataPoint = list(testData.values())[0][:3]

    k = 6

    labels, result = KNN(learining_dataset, dataPoint, k)

    print(labels, result, sep='\n')