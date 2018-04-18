#-*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import json
import pprint
from pythonapi import anno_tools
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

def getData():
    '''获得bbox数据'''
    ALL = []
    with open('train.jsonl') as f:
        lines = f.read().splitlines()
    with open('val.jsonl') as f:
        lines += f.read().splitlines()
    for line in lines:
        anno = json.loads(line.strip())
        for instance in anno_tools.each_char(anno):
            if not instance['is_chinese']:
                continue
            ALL.append(instance['adjusted_bbox'][2:])
    return ALL
def transformData(data):
    return [x[0]/x[1] for x in data]

def getSquare(data):
    return [x[0]*x[1] for x in data]

def kMeans(data, K):
    '''实现聚类过程'''
    print('number of K:', K)
    pred = KMeans(n_clusters=K).fit_predict(data)
    return pred


if __name__=="__main__":
    time_start=time.time()
    data = getData()
    time_end=time.time()
    print('getData cost:', time_end-time_start)
    ratio = transformData(data)
    print(kMeans(ratio, 3))
    