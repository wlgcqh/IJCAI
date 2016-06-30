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
def metric(real_path,predict_path):
    real_file = open(real_path)
    predict_file = open(predict_path)
    real_dict = dict()
    for line in real_file.readlines():
        user_id,location_id,merchant_ids = line.strip().split(',')
        merchant_ids = merchant_ids.split(':')
        for merchant_id in merchant_ids:
            real_dict.setdefault(merchant_id,set())
            real_dict[merchant_id].add((user_id,location_id))
    
    predict_dict = dict()
    for line in predict_file.readlines():
        user_id,location_id,merchant_ids = line.strip().split(',')
        merchant_ids = merchant_ids.split(':')
        for merchant_id in merchant_ids:
            predict_dict.setdefault(merchant_id,set())
            predict_dict[merchant_id].add((user_id,location_id))

    merchant_info_log = open(parent_path+'/data/datasets/ijcai2016_merchant_info')
    merchant_info_dict = dict()
    for line in merchant_info_log.readlines():
        merchant_id,budget = line.strip().split(',')[:2]
        merchant_info_dict[merchant_id] = int(budget)

    s = 0
    pre = 0
    rec = 0
    for merchant in real_dict.keys():
        predict_set = real_dict[merchant]
        real_set = predict_dict.get(merchant,set())
        budget = merchant_info_dict[merchant]

        s += min(budget,len(predict_set&real_set))
        pre += len(predict_set)
        rec += min(budget,len(real_set))
    P = float(s)/pre
    R = float(s)/rec

    F1 = (2*P*R)/(P+R)
    print P,R,F1

'''   
if __name__ == '__main__':
    metric('/home/qh/tianchi/data/datasets/seg_data/ijcai2016_koubei_test_seg','/home/qh/tianchi/data/datasets/seg_data/submission.csv')
'''