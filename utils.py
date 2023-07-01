import os
import json
import traceback
import random
import numpy as np
import copy
import torch

import sys
from sklearn.metrics import jaccard_score


def compute_jaccard(pred_list, trues_list):
    sim=[]
    for pred,true in zip(pred_list,trues_list):
        if true.norm(1)!=0:#如果当天没有用药，则跳过jscore的计算  norm1 表示1范数即绝对值之和，此处用于判断1的个数
            j = jaccard_score(true, pred)
            # assert not np.isnan(j)
            sim.append(j)
    # print('time is {:d}, valid is {:d}'.format(pred_list.shape[0],len(sim)))
    j_score=np.nanmean(sim)
    # assert not np.isnan(j_score)
    return j_score


# y_prob: [[0.7,0.11,0.09,0.1]*]
# y: [0,2,1,3]
# k: 前k个概率值最大的预测
def acck(pred_list,trues_list, k):
    count=0
    day=0
    for pred,true in zip(pred_list,trues_list): #遍历天
        if true.norm(1) != 0:
            day += 1
            _,pred_prob=torch.topk(pred,k)
            # print(pred_prob)
            pred_prob=pred_prob.view(-1).tolist()
            # pred_prob=topk_max_index(pred,k) #该天概率topk药品
            true_lable=true.nonzero().view(-1).tolist() #当天所使用的的药品index
            # print(true_lable)
            intersect=set(pred_prob).intersection(set(true_lable))
            if len(intersect)!=0:
                count+=1
    if day == 0:
        acck = 0
    else:
        acck = float(count)/float(day)
    return acck



def acck_rl(pred_list,trues_list, k):
    count=0
    if trues_list.norm(1) != 0:
        _,pred_prob=torch.topk(pred_list,k) #该天概率topk药品
        pred_prob=pred_prob.tolist()
        true_lable=trues_list.nonzero().view(-1).tolist() #当天所使用的的药品index
        intersect=set(pred_prob).intersection(set(true_lable))
        if len(intersect)!=0:
            count+=1

    acck = float(count)
    return acck

def mywritejson(save_path,content):
    content = json.dumps(content,indent=4,ensure_ascii=False)
    with open(save_path,'w') as f:
        f.write(content)

def myreadjson(load_path):
    with open(load_path,'r') as f:
        return json.loads(f.read())
    
def reward_cal(drug_list,ddi_mat):
    tmp_reward=0
    if drug_list.norm(1) != 0:
        drug_list=drug_list.tolist()
        # print(drug_list)
        drug = [i for i, e in enumerate(drug_list) if e!=0 ]
        # print(drug)
        pos=0
        neg=0
        pred_drug_num=len(drug)
        for i in range(0,pred_drug_num):
            for j in range(i,pred_drug_num):
                if (drug[i],drug[j]) in ddi_mat:
                    if ddi_mat[i,j] == 1.0:
                        pos+=1
                    else:
                        neg+=1
        # print(pos)
        # print(neg)
        tmp_reward = (0.002 * pos - 0.0025 * neg) / (pred_drug_num * pred_drug_num)
        # print(tmp_reward)
    return tmp_reward
