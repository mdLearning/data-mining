#encoding=utf-8
'''
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    print('Start read data...')

    time_1 = time.time()

    raw_data = pd.read_csv('./data/train.csv', header=0)
    data = raw_data.values

    features = data[::, 1::]
    labels = data[::, 0]

    #随机选取33%数据作为测试集，剩余为训练集
    train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33,random_state=0)

    time_2 = time.time()

    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training...')
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(train_features,train_labels)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()

    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels,test_predict)

    print('The accuracy score is %f' % score)
'''
from numpy import *
import numpy as np

# 三大步骤：
'''
1、特征的选择：标准：总方差最小
2、回归树的生成：停止划分的标准
3、剪枝：
'''
#导入数据集
def loadData(fileName):
    dataSet = []
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split('\t')
        theLine = map(float,curLine)
        dataSet.append(theLine)
    return dataSet

# 特征选择：输入：       输出：最佳特征、最佳划分值
'''
1、选择标准
遍历所有的特征Fi：遍历每个特征的所有特征值Zi；找到Zi,划分后总的方差最小
停止划分的条件：
1、当前数据集中的标签相同，返回当前的标签
2、划分前后的总方差差距很小，数据不划分，返回的属性为空，返回的最佳划分值为当前所有标签的均值。
3、划分后的左右两个数据集的样本数量较小，返回的属性为空，返回的最佳划分值为当前所有标签的均值。
当划分的数据集满足上述条件之一，返回的最佳划分值作为叶子节点；
当划分后的数据集不满足上述要求时，找到最佳划分的属性，及最佳划分特征值
'''
#计算方差
def GetAllVar(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]

#根据给定的特征 特征值划分数据集
def dataSplit(dataSet, feature, featNumber):
    dataL = dataSet[nonzero(dataSet[:, feature] > featNumber)[0], :]
    dataR = dataSet[nonzero(dataSet[:, feature] <= featNumber)[0], :]
    return dataL, dataR

#特征划分
def choseBestFeature(dataSet, op = [1,4]):
    if len(set(dataSet[:, -1].T.tolist()[0]))==1:
        regLeaf = mean(dataSet[:, -1])
        return None, regLeaf
    Serror = GetAllVar(dataSet)
    BestFeature = -1; BestNumber = 0; lowError = inf
    m, n = shape(dataSet)#m个样本， n-1 个特征
    for i in range(n-1):#遍历每个特征直
        for j in set(dataSet[:, i].T.tolist()[0]):
            dataL, dataR = dataSplit(dataSet, i, j)
            # 如果所给的划分后的数据集中样本数目甚少，则直接跳出
            if shape(dataR)[0]<op[1] or shape(dataL)[0]<op[1]: continue
            temError = GetAllVar(dataL) + GetAllVar(dataR)
            if temError < lowError:
                lowError  =temError; BestFeature  = i; BestNumber = j
    # 停止条件 2   如果所给的数据划分前后的差别不大，则停止划分
    if Serror - lowError <op[0]:
        return None, mean(dataSet[:, -1])
    dataL,dataR  =dataSplit(dataSet, BestFeature, BestNumber)
    # 停止条件 3
    if shape(dataR)[0] < op[1] or shape(dataL)[0] < op[1]:
        return None, mean(dataSet[:, -1])
    return BestFeature,BestNumber
#决策树生成
def createTree(dataSet, op=[1,4]):
    bestFeat, bestNumber = choseBestFeature(dataSet, op)
    if bestFeat == None: return bestNumber
    regTree = {}
    regTree['spInd'] = bestFeat
    regTree['spVal'] = bestNumber
    dataL, dataR = dataSplit(dataSet, bestFeat, bestNumber)
    regTree['left'] = createTree(dataL,op)
    regTree['right'] = createTree(dataR,op)
    return regTree

# 后剪枝操作
# 用于判断所给的节点是否是叶子节点
def isTree(Tree):
    return (type(Tree).__name__ == 'dict')

# 计算两个叶子节点的均值
def getMean(Tree):
    if isTree(Tree['left']): Tree['left'] = getMean(Tree['left'])
    if isTree(Tree['right']):Tree['right'] = getMean(Tree['right'])
    return (Tree['left'] + Tree['right']/2.0)

# 后剪枝
def pruneTree(Tree, testData):
    if shape(testData)[0] == 0 : return getMean(Tree)
    if isTree(Tree['left']) or isTree(Tree['right']):
        dataL, dataR = dataSplit(testData, Tree['spInd'], Tree['spVal'])
    if isTree(Tree['left']):
        Tree['left'] = pruneTree(Tree['left'], dataL)
    if isTree(Tree['right']):
        Tree['right'] = pruneTree(Tree['right'], dataR)
    if not isTree(Tree['left']) and not isTree(Tree['right']):
        dataL, dataR  =dataSplit(testData,Tree['spInd'],Tree['spVal'])
        errorNoMerge = sum(power(dataL[:,-1] - Tree['left'], 2))+sum(power(dataR[:, -1]-Tree['right'],2))
        leafMean = getMean(Tree)
        erroeMerge = sum(power(testData[:, -1] - leafMean,2))
        if errorNoMerge > erroeMerge:
            print('the leaf merge')
            return leafMean
        else:
            return Tree
    else:
        return Tree
#预测
def forecastSample(Tree, testData):
    if not isTree(Tree): return float(Tree)
    # print"选择的特征是：" ,Tree['spInd']
    # print"测试数据的特征值是：" ,testData[Tree['spInd']]
    if testData[0, Tree['spInd']] > Tree['spVal']:
        if isTree(Tree['left']):
            return forecastSample(Tree['left'], testData)
        else:
            return float(Tree['left'])
    else:
        if isTree(Tree['right']):
            return forecastSample(Tree['right'], testData)
        else:
            return float(Tree['right'])
def TreeForecast(Tree, testData):
    m = shape(testData)[0]
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = forecastSample(Tree, testData[i])

    return y_hat
if __name__ == '__main__':
    print('start...')
    dataMat = loadData('ex2.txt')
    dataMat = mat(dataMat)
    # 参数1：剪枝前总方差与剪枝后总方差差值的最小值；
    # 参数2：将数据集划分为两个子数据集后，子数据集中的样本的最少数量
    op = [1,6]
    theCreateTree = createTree(dataMat, op)
    #测试数据
    dataMat2 = loadData('ex2test.txt')
    dataMat2 = mat(dataMat2)
    # thePruneTree =  pruneTree(theCreateTree, dataMat2)
    # print"剪枝后的后树：\n",thePruneTree
    y = dataMat2[:, -1]
    y_hat = TreeForecast(theCreateTree, dataMat2)
    print(corrcoef(y_hat,y,rowvar=0)[0, 1])
