# -*- coding: utf-8 -*-
__author__ = 'qh'
import os
import random
import csv
import pickle
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
#计算用户间的相似度    
def UserSimikarity(train):
    item_users = dict()
    for u,items in train.items():
        for i in items:
            item_users.setdefault(i,set())
            item_users[i].add(u)

    C= dict()
    N = dict()
    print len(item_users.keys())
    index = 0
    for i,users in item_users.items():
        for u in users:
            N[u] = 0
            for v in users:
                C[u] = {}
                C[u][v] = 0
    print 'init done!'           
        
    for i,users in item_users.items():
        for u in users:
            
            N[u] +=1
            for v in users:
                if u == v:
                    continue
                
                C[u][v] += 1
        if index%100 == 0:
            print index
        index+=1
    print len(C.keys())
    W = dict()
    index = 0
    for u,related_users in C.items():
        W.userSimBest.setdefault(u,dict())
        for v,cuv in related_users.items():
            W.setdefault(u,dict())
            W[u][v] = cuv/math.sqrt(N[u]*N[v])
        if index%10000 == 0:
            print index
        index+=1
    print W
    persistence(W,'user_Sim.txt')
    return W

def Recommend(user,train,W,k = 8,nitem = 40):
    rank = dict()
    interacted_items = train.get(user,{})
    for v ,wuv in sorted(W[user].items(),key = lambda x : x[1],reverse = True)[0:k]:
        for i , rvi in train[v].items():
            if i in interacted_items:
                continue
            rank.setdefault(i,0)
            rank[i] += wuv
    return dict(sorted(rank.items(),key = lambda x :x[1],reverse = True)[0:nitem])

#生成提交结果
def Submission():
    koubei_test_log = open(parent_path+'/data/datasets/ijcai2016_koubei_test')
    Dict = location_merchant()

    submission = open(parent_path+'/data/datasets/submission.csv','w')
    submission_log = csv.writer(submission)
    for line in koubei_test_log.readlines():
        user_id,location_id = line.strip().split(',')
        #print user_id,location_id
        merchant_ids = Dict[location_id]
        #print merchant_ids,len(merchant_ids)
        if len(merchant_ids) < 10:
            recommandMerchants = merchant_ids
        else:
            random.shuffle(merchant_ids)
            #print merchant_ids
            recommandMerchants = merchant_ids[:10]
        string = ''
        for recommandMerchant in recommandMerchants:
            string = string + recommandMerchant + ':'
        #print string,string[:-1]
        submission_log.writerow([int(user_id), int(location_id),string[:-1]])

if __name__ == '__main__':
    train = readPersistence('user_merchant_dict.txt')
    UserSimikarity(train)