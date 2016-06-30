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
#位置与商家的映射
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
#商家与用户的映射
def merchant_user():
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    Dict = dict()
    for line in koubei_train_log.readlines():
        user_id = line.strip().split(',')[0]
        merchant_id = line.strip().split(',')[1]
        
        Dict.setdefault(merchant_id,{})
        Dict[merchant_id].setdefault(user_id,0)
        Dict[merchant_id][user_id]+=1
    persistence(Dict,'merchant_user_dict.txt')  

    return Dict

def train_data():
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
    train_dict = dict()
    for line in koubei_train_log.readlines():
        user_id,merchant_id,location_id = line.strip().split(',')[:3]
        train_dict.setdefault(user_id,{})
        train_dict[user_id].setdefault(location_id,set())
        train_dict[user_id][location_id].add(merchant_id)
    return train_dict

def taobao_user_brand_dict(buy_weight=2,click_weight=1):
    taobao_log = open(parent_path+'/data/datasets/ijcai2016_taobao')
    Dict = dict()
    for line in taobao_log.readlines():
        user_id = line.strip().split(',')[0]
        cat_id = line.strip().split(',')[3]
        is_buy = int(line.strip().split(',')[4])
        Dict.setdefault(user_id,{})
        Dict[user_id].setdefault(cat_id,0)
        if is_buy == 1:
            Dict[user_id][cat_id]+= buy_weight
        else:
            Dict[user_id][cat_id]+= click_weight
    persistence(Dict,'taobao_user_brand_%d_%d.txt'%(buy_weight,click_weight))
    return Dict
#过滤淘宝日志中的无关用户
def filter_user():
    taobao_user_brand_dict = readPersistence('taobao_user_brand_2_1.txt')
    taobao_userSet = set(taobao_user_brand_dict.keys())
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test')
    Dict = dict()
    for line in koubei_test_log.readlines():
        user_id = line.strip().split(',')[0]
        if user_id in taobao_userSet:
            #Dict[user_id] = taobao_user_brand_dict[user_id]
            vector = np.zeros(72)
            for brand,w in taobao_user_brand_dict[user_id].items():
                vector[int(brand)-1] = int(w)
            Dict[user_id] = vector
    #print Dict,len(Dict.keys()),Dict
    return Dict
'''
def generate_userbrand_matrix():
    Dict = filter_user()
    user_list = []
    user_brand_matrix = np.zeros((len(Dict.keys()),72))
    index = 0
    for u,brands in Dict.items():
        user_list.append(u)
        for brand,w in brands.items(): 
            #print brand,w
            user_brand_matrix[index,int(brand)-1] = int(w)
        index+=1
        #print index
    return user_list,user_brand_matrix
'''
def SVD():
    user_list,user_brand_matrix = generate_userbrand_matrix()
    U,Sigma,VT = linalg.svd(user_brand_matrix)
    print Sigma

#基于口碑训练日志统计最热门的商家信息
def hotMerchant_stat():
    koubei_train_log = open(parent_path+'/data/datasets/ijcai2016_koubei_train')
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
        Dict[merchant_id] = float(num)
    print Dict


    return Dict


#从淘宝日志中比较两用户之间的相似度
def UserSimikarity(train,u1,u2):
    '''
    item_users = dict()
    for u,items in train.items():
        for i in items.keys():
            item_users.setdefault(i,set())
            item_users[i].add(u)

    C= dict()
    N = dict()
    #print item_users
    print len(item_users.keys())
    index = 0             
        
    for i,users in item_users.items():
        for u in users:
            N.setdefault(u,0)
            N[u] +=1
            for v in users:
                if u == v:
                    continue
                C.setdefault(u,{})
                C[u].setdefault(v,0)
                C[u][v] += 1
        print index
        index+=1
    print len(C.keys())
    W = dict()
    index = 0
    for u,related_users in C.items():
        W.setdefault(u,dict())
        for v,cuv in related_users.items():
            W.setdefault(u,dict())
            W[u][v] = cuv/math.sqrt(N[u]*N[v])
        if index%10000 == 0:
            print index
        index+=1
    print W
    persistence(W,'user_Sim.txt')
    return W 
    
    '''
    '''
    #print 'start!'
    u1_brand = train.get(u1,{})
    u2_brand = train.get(u2,{})
    #print u1_brand,u2_brand
    u1_feature = np.zeros(72)
    u2_feature = np.zeros(72)
    for brand,w in u1_brand.items():
        #print brand,w
        u1_feature[int(brand)-1] = int(w)
    for brand,w in u2_brand.items():
        
        u2_feature[int(brand)-1] = int(w)
    a = mat(u1_feature)
    b = mat(u2_feature)
    #print a,b
    c = dot(a,b.T)/np.linalg.norm(a)/np.linalg.norm(b)
    #print np.matrix.tolist(c)[0][0]
    return 0.5*np.matrix.tolist(c)[0][0]+0.5
    '''
    
    u1_feature = train.get(u1,{})
    u2_feature = train.get(u2,{})
    #print u1_feature,u2_feature
    overLap = np.nonzero(np.logical_and(u1_feature,u2_feature))[0]
    if len(overLap) == 0:
        return 0

    num = float(sum(u1_feature.T*u2_feature))    
    denom = la.norm(u1_feature)*la.norm(u2_feature)
    #print num/denom
    return 0.5+0.5*(num/denom)
    
    
#计算物品间的相似度    
def ItemSimikarity(train):
    C = dict()
    N = dict()
    index = 0             
        
    for i,users in train.items():
        for u in users:
            N.setdefault(u,0)
            N[u] +=1
            for v in users:
                if u == v:
                    continue
                C.setdefault(u,{})
                C[u].setdefault(v,0)
                C[u][v] += 1
        if index%10000 == 0:
            print index
        index+=1
    W = dict()
    index = 0
    for u,related_items in C.items():
        W.setdefault(u,dict())
        for v,cuv in related_items.items():
            W.setdefault(u,dict())
            W[u][v] = cuv/math.sqrt(N[u]*N[v])
        if index%10000 == 0:
            print index
        index+=1
    print W
    persistence(W,'item_Sim.txt')
    return W

def Recommend(user,train,W,merchant_ids,K = 8,nitem = 10):
    rank = dict()
    ru = train.get(user,{})
    #print ru
    #print W.keys()
    canidate_dict = dict()
    for i in ru:
        #print W[i]
        for merchant_id in merchant_ids:
            if W.has_key(i) and W[i].has_key(merchant_id):
                canidate_dict[merchant_id] = W[i][merchant_id]
            else:
                canidate_dict[merchant_id] = 0

    for i in ru:
        #print sorted(W[i].items(),key = lambda x : x[1],reverse = True)[0:K]

        for j,wj in canidate_dict.items():
            if j in ru:
                continue
            rank.setdefault(j,0)
            rank[j] += wj
    ##print sorted(rank.items(),key = lambda x :x[1],reverse = True)
    return dict(sorted(rank.items(),key = lambda x :x[1],reverse = True)[0:nitem])

#生成提交结果
def Submission():
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test')
    Dict = location_merchant()

    submission = open(parent_path+'/data/datasets/submission.csv','w')
    submission_log = csv.writer(submission)

    #train = readPersistence('user_merchant_dict.txt')
    #W = readPersistence('item_Sim.txt')
    train_dict = train_data()

    taobao_user_brand_dict = readPersistence('SVD_dict.txt')
    taobao_userSet = set(taobao_user_brand_dict.keys())

    merchant_user_dict = readPersistence('merchant_user_dict.txt')
    hotMerchant_dict = hotMerchant_stat()
    index = 0
    count = 0
    print 'start!'
    for line in koubei_test_log.readlines():
        user_id,location_id = line.strip().split(',')
        #print user_id,location_id
        merchant_ids = Dict[location_id]
        #推荐列表大于10才需要筛选
        if len(merchant_ids) > 6:
            #不在口碑的训练日志里
            '''
            if train.get(user_id)==None:
                #在淘宝的日志中
                merchantDict = dict()
                if user_id in taobao_userSet:         
                    for merchant_id in merchant_ids:
                        users = merchant_user_dict.get(merchant_id,{})
                        sim = 0
                        for u in users:
                            if u not in taobao_userSet:
                                continue
                            sim_2users = UserSimikarity(taobao_user_brand_dict,user_id,u)
                            #print float(sim_2users),type(sim_2users)
                            sim += merchant_user_dict[merchant_id][u]*float(sim_2users)
                        merchantDict[merchant_id] = sim
            
                #未在淘宝日志中出现的新用户，推荐最热门店铺
                else:
                    for merchant_id in merchant_ids:
                        merchantDict[merchant_id] = hotMerchant_dict[merchant_id]
                    #print merchantDict
                recommandMerchants = dict(sorted(merchantDict.items(),key = lambda x :x[1],reverse = True)[:5]).keys()   
                count+=1 
            else:
                #print 'recommend'
                recommandMerchants = Recommend(user_id,train,W,merchant_ids).keys()
            '''
            merchantDict = dict() 
            for merchant_id in merchant_ids:
                merchantDict[merchant_id] = hotMerchant_dict.get(merchant_id,0)

            try:
                merchant_set = train_dict[user_id][location_id]
                if len(merchant_set)<5:
                    #print len(merchant_set)
                    his_list = list(merchant_set)
                    #hot_list = dict(sorted(merchantDict.items(),key = lambda x :x[1],reverse = True)[:(5-len(merchant_set))]).keys()
                    #his_list.extend(hot_list)
                    recommandMerchants = list(set(his_list))
                    #print his_list,hot_list,recommandMerchants
                else:
                    recommandMerchants = list(merchant_set)    
                index+=1          
            except:
                
                recommandMerchants = dict(sorted(merchantDict.items(),key = lambda x :x[1],reverse = True)[:5]).keys()
        else:
            recommandMerchants = merchant_ids
        #print merchant_ids
        #print recommandMerchants
        #print '------------------'
        string = ''
        for recommandMerchant in recommandMerchants:
            string = string + recommandMerchant + ':'
        #print string,string[:-1]
        submission_log.writerow([int(user_id), int(location_id),string[:-1]])

        index+=1
        if index%10000 == 0:
            print index
            index+=1
        #print index

if __name__ == '__main__':
    '''
    train = readPersistence('user_merchant_dict.txt')
    W = readPersistence('item_Sim.txt')
    Dict = location_merchant()
    #merchant_ids = Dict[location_id]
    user_id = '1924736'
    print Recommend(user_id,train,W,Dict['278'])
    
    train = filter_user()
    UserSimikarity(train)
    '''
    #SVD()
    Submission()