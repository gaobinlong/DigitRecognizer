# -*- coding: utf-8 -*-

from numpy import *
import operator
import sys
import csv


def loadTrainData():
    l = []
    with open('../data/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    l = array(l)
    print l
    label = l[:, 0]
    data = l[:, 1:]
    print toInt(label)
    return normizing(toInt(data)), toInt(label)


def toInt(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n),dtype = int8)
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def normizing(array):
    m, n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


def loadTestData():
    l = []
    with open('../data/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    data = array(l)
    return normizing(toInt(data))


def knn(record, dataset, labels, k):
    record = mat(record)
    dataset = mat(dataset)
    labels = mat(labels)
    datasetSize = dataset.shape[0]
    diff = tile(record, (datasetSize, 1)) - dataset
    sqrtDiff = array(diff)**2
    sqrtDis = sqrtDiff.sum(axis=1)
    dis = sqrtDis**0.5
    sortDis = dis.argsort()
    classfier = {}
    for i in xrange(k):
        voteLabel = labels[0, sortDis[i]]
        classfier[voteLabel] = classfier.get(voteLabel, 0) + 1
    sortClassfier = sorted(classfier.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortClassfier[0][0]


def saveResult(result):
    with open('../data/result.csv', 'wb') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageId', 'Label'])
        for i in result:
            writer.writerow([i+1, result[i]])


def handingWriting():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    m, n = shape(testData)
    resultList = {}
    for i in xrange(m):
        result = knn(testData[i], trainData, trainLabel, 5)
        resultList[i] = result
    saveResult(resultList)

if __name__ == '__main__':
    handingWriting()
