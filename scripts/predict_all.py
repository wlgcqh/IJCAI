# -*- coding: utf-8 -*-
__author__ = 'qh'
import os
import random
import csv
import pickle
import math
import numpy as np
import time
from numpy import *
from numpy import linalg as la
import evaluationMetric
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

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
def train_data():
    koubei_train_log = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_train_seg')
    train_dict = dict()
    for line in koubei_train_log.readlines():
        user_id,merchant_id,location_id = line.strip().split(',')[:3]
        train_dict.setdefault(user_id,{})
        train_dict[user_id].setdefault(location_id,set())
        train_dict[user_id][location_id].add(merchant_id)
    return train_dict

def test_data(taobao_user_dict,merchant_dict,user,merchants,clf,scaler):
    t0 = time.time()
    feature_matrix = []
    vector = np.zeros(83)
    _dict = taobao_user_dict.get(user,None)
    isTaoBaoUser = 1
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
    for merchant_id in merchants:
        negtive_vector = np.zeros(83)
        negtive_vector[:79] = vector[:79]
        _merchant_dict = merchant_dict[merchant_id]
        negtive_vector[79:83] = _merchant_dict['budget'],_merchant_dict['buy_Num'],_merchant_dict['rate'],_merchant_dict['location_Num']
        feature_matrix.append(negtive_vector)
    '''
    t1 = time.time()
    print("time :  %.3f s"% (t1-t0))
    normed_test = scaler.transform(np.array(feature_matrix))
    t2 = time.time()
    print("time :  %.3f s"% (t2-t1))
    #print clf.predict(normed_test)  
    probs =  clf.predict_proba(normed_test)[:,1]
    t3 = time.time()
    print("time :  %.3f s"% (t3-t2))
    Dict = dict()
    for merchant,prob in zip(merchants,probs):
        Dict[merchant] = prob
    
    List = dict(sorted(Dict.items(),key = lambda x :x[1],reverse = True)[:5]).keys()
    
    print '-----------------------------'
    '''
    return np.array(feature_matrix)


#生成提交结果
def Submission():
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test').readlines()
    Dict = location_merchant()

    submission = open(parent_path+'/data/datasets/submission.csv','w')
    submission_log = csv.writer(submission)

    hotMerchant_dict = hotMerchant_stat(parent_path+'/data/datasets/ijcai2016_koubei_train')
    train_dict = train_data()
    #taobao_user_dict = readPersistence('user_dict.txt')
    merchant_dict = merchant_feature()
    index = 0

    #trainData = np.load('user_feature_all.npy')
    print 'Normalized!'
    #minMaxScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #scaler = minMaxScaler.fit(trainData)
    print 'clf'
    #clf = joblib.load('/home/qh/tianchi/model/rfc.pkl')

    print 'start!'
    test_feature = []
    '''
    for line in koubei_test_log:
        user_id,location_id = line.strip().split(',')
        #print user_id,location_id
        merchant_ids = Dict[location_id]
        #推荐列表大于10才需要筛选
        
        
        feature = test_data(taobao_user_dict,merchant_dict,user_id,merchant_ids,clf,scaler)
        for i in feature:
            test_feature.append(i)
        
  
        if index%10000==0:
            print index
        index+=1
        
    normed_test = scaler.transform(np.array(test_feature))
    print 'predict'
    probs =  clf.predict_proba(normed_test)[:,1]
    print probs[:10],probs.shape
    np.save('prob_all.npy',probs)

    '''
    probs = np.load('prob_all.npy')
    i = 0
    for line in koubei_test_log:
        user_id,location_id = line.strip().split(',')
        #print user_id,location_id
        merchant_ids = Dict[location_id]
        _dict = dict()
        lens = len(merchant_ids)
        for m in merchant_ids:
            _dict[m] = probs[i]
            i+=1
        
        #recommandMerchants = dict(sorted(_dict.items(),key = lambda x :x[1],reverse = True)[:1]).keys()

        
        #不在口碑的训练日志里
        _count = 0
        merchantDict = dict() 
        for merchant_id in merchant_ids:
            merchantDict[merchant_id] = hotMerchant_dict.get(merchant_id,0)
         
        #推荐最热门商品
        try:
            merchant_set = train_dict[user_id][location_id]
            #print len(merchant_set)
            
            recommandMerchants = list(merchant_set)
            index+=1           
        except:
            for k,v in _dict.items():
                if v > 0.8:
                    _count+=1  
            recommandMerchants = dict(sorted(_dict.items(),key = lambda x :x[1],reverse = True)[:2]).keys()
          
        #recommandMerchants = dict(sorted(merchantDict.items(),key = lambda x :x[1],reverse = True)[:1]).keys()
        print _count
        string = ''
        for recommandMerchant in recommandMerchants:
            string = string + recommandMerchant + ':'
        #print string,string[:-1]
        submission_log.writerow([int(user_id), int(location_id),string[:-1]])
    print i,index
    
    
if __name__ == '__main__':
    
    Submission()
    #evaluationMetric.metric('/home/qh/tianchi/data/datasets/seg_data/ijcai2016_koubei_test_seg','/home/qh/tianchi/data/datasets/seg_data/submission.csv')
