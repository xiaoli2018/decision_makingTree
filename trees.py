# -*-coding:utf-8 -*-
__author__ = 'liheng'
'''
数据挖掘之机器学习算法： 决策树
'''

from math import log

#计算给定的集的香农熵
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

def creatDataSet():
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
    numFeature = len(dataSet[0])-1
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







