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
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

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

#淘宝日志特征提取
def taobao_log_feature():
    #获取口碑测试集的用户
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test')
    test_userSet = set()
    for line in koubei_test_log.readlines():
        test_userSet.add(line.strip().split(',')[0])
    #获取口碑训练集的用户
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    train_userSet = set()
    for line in koubei_train_log.readlines():
        train_userSet.add(line.strip().split(',')[0])

    taobao_log = open(parent_path+'/data/datasets/ijcai2016_taobao')
    Dict = dict()


    index=0
    for line in taobao_log.readlines():
        if index%1000000 == 0:
            print index

        user_id,seller_id,item_id,cat_id,is_buy,time = line.strip().split(',')
        if (user_id not in train_userSet) and  (user_id not in test_userSet):
            continue
        Dict.setdefault(user_id,{})
        Dict[user_id].setdefault('buyNum',0)
        Dict[user_id].setdefault('clickNum',0)       
        Dict[user_id].setdefault('activeDays',[])
        Dict[user_id].setdefault('buyDays',[])
        Dict[user_id].setdefault('brand_click',np.zeros(72))
        Dict[user_id].setdefault('brand_buy',np.zeros(72))

        Dict[user_id]['buyNum']+=int(is_buy)
        Dict[user_id]['clickNum']+=(1-int(is_buy))       
        Dict[user_id]['activeDays'].append(time)
        if int(is_buy)==1:
            Dict[user_id]['buyDays'].append(time)
            Dict[user_id]['brand_buy'][int(cat_id)-1]+=1
        else:
            Dict[user_id]['brand_click'][int(cat_id)-1]+=1

        #print Dict[user_id]
        index+=1   
    #persistence(Dict,'user_dict.txt')
    print len(Dict.keys()) 
#提取店铺特征
def merchant_feature():
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    hotMerchant_dict = dict()
    for line in koubei_train_log.readlines():
        merchant_id = line.strip().split(',')[1]
        hotMerchant_dict.setdefault(merchant_id,0)
        hotMerchant_dict[merchant_id]+=1

    merchant_info_log = open(parent_path+'/data/datasets/ijcai2016_merchant_info')
    Dict = dict()
    for line in merchant_info_log.readlines():
        merchant_id = line.strip().split(',')[0]
        budget = float(line.strip().split(',')[1])
        location_ids = line.strip().split(',')[-1].split(':')
        Dict.setdefault(merchant_id,{})
        Dict[merchant_id]['budget'] = budget
        Dict[merchant_id]['buy_Num'] = hotMerchant_dict[merchant_id]
        Dict[merchant_id]['rate'] = budget/hotMerchant_dict[merchant_id]
        Dict[merchant_id]['location_Num'] = len(location_ids)
    #print Dict
    return Dict

def location_merchant():
    merchant_info_log = open(parent_path+'/data/datasets/ijcai2016_merchant_info')
    Dict = dict()
    for line in merchant_info_log.readlines():
        merchant_id = line.strip().split(',')[0]
        location_ids = line.strip().split(',')[-1].split(':')
        for location_id in location_ids:
            if location_id not in Dict.keys():
                Dict[location_id] = []
            Dict[location_id].append(merchant_id)
                
    return Dict

#基于口碑训练日志统计最热门的商家信息
def hotMerchant_stat(path):
    koubei_train_log = open(path)
    hotMerchant_dict = dict()
    for line in koubei_train_log.readlines():
        merchant_id = line.strip().split(',')[1]
        hotMerchant_dict.setdefault(merchant_id,0)
        hotMerchant_dict[merchant_id]+=1

    merchant_info_log = open(parent_path+'/data/datasets/ijcai2016_merchant_info')
    merchant_budget_dict = dict()
    for line in merchant_info_log.readlines():
        merchant_id = line.strip().split(',')[0]
        budget = line.strip().split(',')[1]
        merchant_budget_dict[merchant_id] = budget

    Dict = dict()
    for merchant_id,num in hotMerchant_dict.items():
        budget = int(merchant_budget_dict.get(merchant_id))
        Dict[merchant_id] = num
    #print Dict
    return Dict

def train_data_generate():
    koubei_train_log = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_train_seg')
    train_dict = dict()
    for line in koubei_train_log.readlines():
        user_id,merchant_id,location_id = line.strip().split(',')[:3]
        train_dict.setdefault(user_id,{})
        train_dict[user_id].setdefault(location_id,set())
        train_dict[user_id][location_id].add(merchant_id)
    return train_dict

#用koubei_train里面的数据制作正负样本 user_merchant的pair
def train_data():
    
    merchant_dict = merchant_feature()
    taobao_user_dict = readPersistence('user_dict.txt')

    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    hotMerchant_dict = hotMerchant_stat(parent_path+'/data/datasets/ijcai2016_koubei_train')
    location_merchant_dict = location_merchant()
    train_dict = train_data_generate()
    Dict = dict()
    feature_matrix = []
    label = []
    index = 0
    print 'start'
    for line in koubei_train_log:
        user_id,merchant_id,location_id,time = line.strip().split(',')
        vector = np.zeros(83)
        
        isTaoBaoUser = 1
        _dict = taobao_user_dict.get(user_id,None)
        if _dict==None:
            isTaoBaoUser = 0
            vector[:79] = 0
        else:
            
            vector[0] = _dict['buyNum']
            vector[1] = _dict['clickNum']
            vector[2] = float(vector[0])/(vector[0]+vector[1])
            vector[3] = len(set(_dict['activeDays']))
            vector[4] = len(set(_dict['buyDays']))
            vector[5] = float(vector[4])/(vector[3]+vector[4])
            vector[6:78] = _dict['brand_click'] + 2*_dict['brand_buy']
            vector[78] = isTaoBaoUser
            
        _merchant_dict = merchant_dict[merchant_id]
        vector[79:83] = _merchant_dict['budget'],_merchant_dict['buy_Num'],_merchant_dict['rate'],_merchant_dict['location_Num']
        feature_matrix.append(vector)
        label.append(1)
        
        merchant_ids = location_merchant_dict[location_id]
        try:
            merchant_set = train_dict[user_id][location_id]
            merchant_ids = list(set(merchant_ids) - set(merchant_set))
        except:
            pass

        merchantDict = dict() 
        for merchant_id1 in merchant_ids:
            merchantDict[merchant_id1] = hotMerchant_dict.get(merchant_id1,0)
        s = dict(sorted(merchantDict.items(),key = lambda x :x[1],reverse = True)[:50]).keys()
        random.shuffle(s)
        negtiveMerchants = s[:10]
        
        for merchant_id2 in negtiveMerchants:
            negtive_vector = np.zeros(83)
            negtive_vector[:79] = vector[:79]
            _merchant_dict1 = merchant_dict[merchant_id2]
            negtive_vector[79:83] = _merchant_dict1['budget'],_merchant_dict1['buy_Num'],_merchant_dict1['rate'],_merchant_dict1['location_Num']
            feature_matrix.append(negtive_vector)
            label.append(0)
        #print merchant_id,merchant_ids
        
        if index%10000 == 0:
            print index,vector
        index+=1
    np.save('user_feature_all.npy',np.array(feature_matrix))
    np.save('label_all.npy',np.array(label))

def train_model():
    trainData = np.load('user_feature_all.npy')
    trainLabels = np.load('label_all.npy')
    print 'Normalized!'
    minMaxScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler = minMaxScaler.fit(trainData)
    normed_train = scaler.transform(trainData)

    rfc = RandomForestClassifier(n_estimators=1000,n_jobs=-1)
    rfc.fit(normed_train,trainLabels)
    joblib.dump(rfc, '/home/qh/tianchi/model/rfc.pkl')
if __name__ == '__main__':
    train_model()