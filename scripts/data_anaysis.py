# -*- coding: utf-8 -*-
__author__ = 'qh'
import os
import random
import csv
import pickle
import numpy as np
dataPath=os.getcwd()
parent_path = os.path.dirname(dataPath)
print parent_path

def taobao_stat():
    taobao_log = open(parent_path+'/data/datasets/ijcai2016_taobao')
    userSet = set()
    sellerSet = set()
    itemSet = set()
    catSet =set()
    buy_Num = 0
    timeStampSet = set()
    buyerSet = set()
    index = 0
    for line in taobao_log.readlines():
        
        userSet.add(line.strip().split(',')[0])
        '''
        sellerSet.add(line.strip().split(',')[1])
        itemSet.add(line.strip().split(',')[2])
        catSet.add(line.strip().split(',')[3])
        timeStampSet.add(line.strip().split(',')[5])
        
        if int(line.strip().split(',')[4]) == 1:
            buyerSet.add(line.strip().split(',')[0])
            buy_Num += 1
        '''
        index+=1 
    #print len(buyerSet),index
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test')
    test_userSet = set()
    for line in koubei_test_log.readlines():
        test_userSet.add(line.strip().split(',')[0])

    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    train_userSet = set()
    for line in koubei_train_log.readlines():
        train_userSet.add(line.strip().split(',')[0])

    count = 0
    for train_user in train_userSet:
        if train_user in userSet:
            count+=1
    print len(userSet),len(train_userSet),len(test_userSet),count

def koubei_train_stat():
    taobao_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    userSet = set()
    merchantSet = set()
    locationSet = set()
    timeStampSet = set()
    index = 0
    c7,c8,c9,c10,c11 = 0,0,0,0,0
    for line in taobao_log.readlines():
        
        userSet.add(line.strip().split(',')[0])
        merchantSet.add(line.strip().split(',')[1])
        locationSet.add(line.strip().split(',')[2])
        timeStampSet.add(line.strip().split(',')[3])
        time = int(line.strip().split(',')[3])
        if time < 20150801:
            c7+=1
        elif time < 20150901:
            c8+=1
        elif time < 20151001:
            c9+=1
        elif time < 20151101:
            c10+=1
        elif time < 20151201:
            c11+=1
        
        index+=1 
    print len(userSet),len(merchantSet),len(locationSet),len(timeStampSet),index
    print c7,c8,c9,c10,c11
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
def user_merchant():
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train').readlines()
    Dict = dict()
    userSet = set()
    index = 0
    for line in koubei_train_log:
        user_id = line.strip().split(',')[0]
        userSet.add(user_id)
    print '1'
    for u in userSet:
        Dict[u] = []
    print '2'
    for line in koubei_train_log:
        user_id = line.strip().split(',')[0]
        merchant_id = line.strip().split(',')[1]
        Dict[user_id].append(merchant_id)
        if index%10000 == 0:
            print index
        index+=1
    persistence(Dict,'user_merchant_dict.txt')

    #print Dict
def train_data():
    koubei_train_log = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_train_seg')
    train_dict = dict()
    for line in koubei_train_log.readlines():
        user_id,merchant_id,location_id = line.strip().split(',')[:3]
        train_dict.setdefault(user_id,{})
        train_dict[user_id].setdefault(location_id,set())
        train_dict[user_id][location_id].add(merchant_id)
    return train_dict

def buy_Num():
    koubei_test_log = open(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_test_seg')
    taobao_user_dict = readPersistence('user_dict.txt')
    print 'start'
    train_dict = train_data()
    array_taobao = []
    array_else = []
    index = 0
    num = 0
    for line in koubei_test_log:
        user_id,location_id,merchant_ids = line.strip().split(',')
        merchant_ids = merchant_ids.split(':')
        
        _dict = taobao_user_dict.get(user_id,None)
        if _dict == None:
            array_else.append(len(merchant_ids))
        else:
            array_taobao.append(len(merchant_ids))
        
        num+=len(merchant_ids)
        index+=1
    print num/float(index)
    
    array_taobao = np.array(array_taobao)
    array_else = np.array(array_else)
    print np.bincount(array_taobao),array_taobao.max(),array_taobao.min(),array_taobao.mean()
    print np.bincount(array_else),array_else.max(),array_else.min(),array_else.mean()
  
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
        Dict.setdefault(merchant_id,{})
        Dict[merchant_id]['budget'] = budget
        Dict[merchant_id]['Num'] = num
    sub_dict = submission_anaysis()    
    for key,val in  sorted(Dict.items(),key = lambda x :x[1]['Num'],reverse = True)[:100]:
        print key,val,sub_dict[key]
    return Dict  

#基于口碑训练日志统计最热门的商家信息
def hotLocationMerchant_stat(path):
    koubei_train_log = open(path)
    hotMerchant_dict = dict()
    for line in koubei_train_log.readlines():
        merchant_id = line.strip().split(',')[1]
        location_id = line.strip().split(',')[2]
        hotMerchant_dict.setdefault(location_id,{})
        hotMerchant_dict[location_id].setdefault(merchant_id,0)
        hotMerchant_dict[location_id][merchant_id]+=1

    #print len(hotMerchant_dict.keys())
    return hotMerchant_dict  
def submission_anaysis():
    submission_log = open(parent_path+'/data/datasets/seg_data/submission.csv')
    Dict = dict()
    for line in submission_log.readlines():
        user_id,location_id,merchant_ids = line.strip().split(',')
        merchant_ids = merchant_ids.split(':')
        for m in merchant_ids:
            Dict.setdefault(m,0)
            Dict[m]+=1
    '''
    hot_dict = hotMerchant_stat(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_train_seg')
    Dict = sorted(Dict.items(),key = lambda x :x[1],reverse = True)[:100]
    for key,val in Dict:
        print key,hot_dict[key],val
    '''
    return Dict

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


if __name__ == '__main__':
    hotLocationMerchant_stat(parent_path+'/data/datasets/seg_data/ijcai2016_koubei_train_seg')
    #submission_anaysis()  
