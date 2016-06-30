# -*- coding: utf-8 -*-
__author__ = 'qh'
import os
import random
import csv
import pickle
import math
import numpy as np
#import scipy.sparse.linalg as la
from numpy import linalg as la
dataPath=os.getcwd()
parent_path = os.path.dirname(dataPath)
print parent_path

def persistence(obj,filePath):
    file = open(filePath,'w')
    _obj = pickle.dumps(obj)
    file.write(_obj)
    file.close()

#读取持久化文件
def readPersistence(filePath):
    file = open(filePath)
    return pickle.loads(file.read())
def filter_user():
    taobao_user_brand_dict = readPersistence('taobao_user_brand_2_1.txt')
    taobao_userSet = set(taobao_user_brand_dict.keys())
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test')
    Dict = dict()
    for line in koubei_test_log.readlines():
        user_id = line.strip().split(',')[0]
        if user_id in taobao_userSet:
            Dict[user_id] = taobao_user_brand_dict[user_id]
            
    #print Dict,len(Dict.keys()),Dict
    return Dict

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
def SVD():
    user_list,user_brand_matrix = generate_userbrand_matrix()
    U,Sigma,VT = la.svds(user_brand_matrix,k=71)
    Sigma = np.array(Sigma)
    U = np.mat(U)
    VT = np.mat(VT)
    '''
    print Sigma
    sig2 = Sigma**2
    print sum(sig2)
    print sum(sig2)*0.9
    print sum(sig2[-10:])/sum(sig2)
    print sum(sig2[-15:])/sum(sig2)
    print sum(sig2[-20:])/sum(sig2)
    '''
    Sig20 = np.mat(np.eye(20)*Sigma[-20:])
    print (user_brand_matrix*(VT[-20:,:].T)*Sig20).shape
    SVD_matrix = user_brand_matrix*(VT[-20:,:].T)*Sig20.I
    Dict = dict()
    index = 0
    for u in user_list:
        Dict[u] = np.array(SVD_matrix[index]).reshape(20)
        index +=1
    persistence(Dict,'SVD_dict.txt')
    #print Dict

#从淘宝日志中比较两用户之间的相似度
def UserSimikarity(train):
    userSet = train.keys()
    Dict = dict()
    index = 0
    for u1 in userSet:
        Dict.setdefault(u1,{})
        for u2 in userSet:
            if u1==u2:
               continue
            u1_feature = train.get(u1,{})
            u2_feature = train.get(u2,{})
            #print u1_feature,u2_feature
            overLap = np.nonzero(np.logical_and(u1_feature,u2_feature))[0]
            if len(overLap) == 0:
                Dict[u1][u2] = 0
                continue
            num = float(sum(u1_feature.T*u2_feature))
            denom = la.norm(u1_feature)*la.norm(u2_feature)
            #print num/denom
            Dict[u1][u2] = num/denom
        print index
        index+=1
    persistence(Dict,'taobao_user_Sim.txt')
       
if __name__ == '__main__':
    train = readPersistence('SVD_dict.txt')
    UserSimikarity(train)
