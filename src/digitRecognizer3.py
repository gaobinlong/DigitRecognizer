# -*- coding: utf-8 -*-

# use logistic regression algorithm
from numpy import *
import csv


def loadTrainData():
    l = []
    with open('../data/train1.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    l = array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return normizing(toInt(data)), toInt(label)


def classLable(lable, tag):
    newLable = copy(lable)
    i = 0
    length = len(lable[0])
    for i in xrange(length):
        if lable[0][i] == tag:
            newLable[0][i] = 1
        else:
            newLable[0][i] = 0
    return newLable


def toInt(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n), dtype=int8)
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
    with open('../data/test1.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    data = array(l)
    return normizing(toInt(data))


def sigmoid(intX):
    return 1.0 / (1 + exp(-intX))


def gradAscent(dataMatIn, classLabels,alpha, maxCycles):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in xrange(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def classify(testData, weight):
    testData = mat(testData)
    h = sigmoid(testData * weight)
    m = len(h)
    resultList = {}
    for i in xrange(m):
        if int(h[i][0]) > 0.5:
            resultList[i] = 1
        else:
            resultList[i] = 0
    return resultList


def saveResult(resultList, tag, finalResult):
    m = len(resultList)
    for i in xrange(m):
        if resultList[i] == 1:
            finalResult[i] = tag


def writeFile(finalResult):
    with open('../data/result3.csv', 'wb') as file:
        writer = csv.writer(file)
        for i in finalResult:
            writer.writerow([i + 1, finalResult[i]])


def digitRecognizer(alpha=0.07, maxCycles=10):
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    m, n = shape(testData)
    finalResult = {}
    for i in xrange(10):
        newLable = classLable(trainLabel, i)
        weight = gradAscent(trainData, newLable, alpha, maxCycles)
        resultList = classify(testData, weight)
        saveResult(resultList, i, finalResult)
    writeFile(finalResult)

if __name__ == '__main__':
    digitRecognizer()
