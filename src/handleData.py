# -*- coding: utf-8 -*-

import csv
import numpy
import sys

TRAIN_DATA_FILE = '../data/train1.csv'
TEST_DATA_FILE = '../data/test1.csv'
RESULT_FILE = '../data/submmit.csv'
PIXEL_LENGTH = 784

imagePixelMap = {}
imagePixelCountMap = {}


def handle_train_data():
    reader = csv.reader(open(TRAIN_DATA_FILE, 'rb'))
    count = 0
    for item in reader:
        if count == 0:
            count = count + 1
            continue
        else:
            li = [int(x) for x in item if x]
            num = li[0]
            if num not in imagePixelMap:
                imagePixelMap[num] = []
                imagePixelCountMap[num] = 0
                for i in range(1, PIXEL_LENGTH + 1):
                    imagePixelMap[num].append(li[i])
            else:
                imagePixelCountMap[num] = imagePixelCountMap[num] + 1
                imagePixel = imagePixelMap[num]
                for i in range(0, PIXEL_LENGTH):
                    imagePixel[i] = (imagePixel[i] * (count - 1) + li[i + 1]) / float(count)
        count = count + 1


def handle_test_data():
    reader = csv.reader(open(TEST_DATA_FILE, 'rb'))
    writer = csv.writer(open(RESULT_FILE, 'wb'))
    writer.writerow(['ImageId', 'Label'])
    count = 0
    for item in reader:
        if count == 0:
            count = count + 1
            continue
        else:
            lint = [int(x) for x in item if x]
            index = cal_distance(lint)
            writer.writerow([count, index])
            count = count + 1


def cal_distance(record):
    minDis = sys.maxint
    index = 0
    for key in imagePixelMap:
        center = imagePixelMap[key]
        dis = 0
        for i in range(0, PIXEL_LENGTH):
            iDis = center[i] - record[i]
            dis = iDis**2 + dis
        dis = dis**0.5
        if dis < minDis:
            minDis = dis
            index = key
    imagePixel = imagePixelMap[index]
    oldlen = imagePixelCountMap[index]
    imagePixelCountMap[index] = imagePixelCountMap[index] + 1
    newlen = imagePixelCountMap[index]
    for j in range(0, PIXEL_LENGTH):
        imagePixel[j] = (imagePixel[j] * oldlen + record[j]) / (newlen)
    return index
