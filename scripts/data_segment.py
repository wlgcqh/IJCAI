# -*- coding: utf-8 -*-
__author__ = 'qh'
import os
import random
import csv
import pickle
import math
import numpy as np
from numpy import *
from numpy import linalg as la
dataPath=os.getcwd()
parent_path = os.path.dirname(dataPath)
print parent_path
#将指定内容持久化到文件中
def persistence(obj,filePath):
    file = open(filePath,'w')
    _obj = pickle.dumps(obj)
    file.write(_obj)
    file.close()

#读取持久化文件
def readPersistence(filePath):
    file = open(filePath)
    return pickle.loads(file.read())

def seg():
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    train_seg = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_train_seg','w')
    test_seg = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_test_seg','w')
    test_seg_sub = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_test_seg_sub','w')

    Dict = dict()
    for line in koubei_train_log.readlines():
        user_id,merchant_id,location_id,time = line.strip().split(',')
        if int(time) < 20151101:
            train_seg.write(line)
        else:
            Dict.setdefault(user_id,{})
            Dict[user_id].setdefault(location_id,[])
            Dict[user_id][location_id].append(merchant_id)
    #print Dict
    for user,val in Dict.items():
        for location,merchants in val.items():
            s=''
            for merchant in merchants:
                s=s+merchant+':'
            test_seg.write(user+','+location+','+s[:-1])
            test_seg.write('\n')
            test_seg_sub.write(user+','+location)
            test_seg_sub.write('\n')
    train_seg.close()
    test_seg.close()
if __name__ == '__main__':
    seg()
