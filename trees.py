# -*-coding:utf-8 -*-
__author__ = 'liheng'
'''
数据挖掘之机器学习算法： 决策树
'''

from math import log
import operator


#计算给定的集的香农熵
#香农熵暂且简单的理解为信息的混乱程度
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        #计算出每种类别出现的次数
        if currentLabel not in labelCounts.keys():
            #如果没有此类别就进行扩展
            labelCounts[currentLabel] = 0
        #每次对出现的类别进行加一
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #通过每种类别出现的频率计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#提取符合要求的数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeaVec = featVec[:axis]
            reduceFeaVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeaVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureTopSplit(dataSet):
    numFeature = len(dataSet[0])-1    #将最后的类别标签提取出去
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bastFeature = -1
    for i in range(numFeature):
        #创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        #计算每种划分方式的香农熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        #计算划分前后的信息增益，信息增益越大表示划分的数据对整体数据的影响越大
        infoGain  = baseEntropy - newEntropy
        #寻找使信息增益最大的数据
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#多数表决程序
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]   #返回的是出现频率最高的标签

#构建决策树
def creatTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #当类别是完全相等的时候停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果只有一种元素，返回出现的次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureTopSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree













