import sys

import os
import sys
import time
import numpy as np
from sklearn.metrics import precision_score,jaccard_score,recall_score,f1_score
import random
import json
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.multiprocessing import Pool
import copy
import time
import scipy.sparse as sp

import warnings

warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
torch.multiprocessing.set_sharing_strategy('file_system')

import data_loader
import model
import utils
import parse

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

args = parse.args
model_dir = args.model_dir

ddi = pd.read_csv('../../data/drop_ddi.csv').values.tolist()
drug_num = 307
ddi_mat = sp.dok_matrix((drug_num, drug_num), dtype=np.float32)
for x in ddi:
    ddi_mat[x[0], x[1]] = x[2]

def soft_update(net, target_net):
    for param_target, param in zip(target_net.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - args.tau) + param.data * args.tau)

def FedAvg(usr_model_weights, avg_weight):
    weight_avg = copy.deepcopy(usr_model_weights[0])
    for k in weight_avg.keys():
        weight_avg[k] = weight_avg[k] * avg_weight[0]
        for i in range(1, len(usr_model_weights)):
            weight_avg[k] += usr_model_weights[i][k] * avg_weight[i]
    return weight_avg


def getWeightOfModel(model):
    weights = []
    for key, value in model.items():
        weights.append(value)
    return torch.cat([x.flatten() for x in weights])


def train_srl_run(epoch, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, train_data, device):
    start = time.time()
    actor.train()
    critic.train()

    actor_loss_list = []
    critic_loss_list = []

    j_score_list = []
    precision_list=[]
    recall_list=[]
    f1_list=[]

    for data in train_data:
        state_crt, state_nxt, action_crt, action_nxt, mask_nxt, reward = data[:6]
        state_crt, state_nxt, action_crt, action_nxt, mask_nxt, reward = \
            state_crt.to(device), state_nxt.to(device), action_crt.to(device), action_nxt.to(device), mask_nxt.to(device), reward.to(device)

        ori_action_nxt_prob = actor(state_crt)
        zero = torch.zeros_like(ori_action_nxt_prob)
        one = torch.ones_like(ori_action_nxt_prob)

        ori_action_nxt_prob = torch.where(ori_action_nxt_prob >= 0.5, one, ori_action_nxt_prob)
        ori_action_nxt_prob = torch.where(ori_action_nxt_prob < 0.5, zero, ori_action_nxt_prob)

        idx = torch.argmax(mask_nxt, 1)
        idx[idx == 0] = args.length
        shape = ori_action_nxt_prob.shape[0]

        for b in range(shape):
            pred_action_nxt = ori_action_nxt_prob.data.cpu()[b, :idx[b], :]
            for day in range(0,idx[b]):
                drug_list=pred_action_nxt[day,:].view(-1)
                reward_b=utils.reward_cal(drug_list,ddi_mat)
                reward[b][day] += reward_b

        target_q_values = target_critic(state_nxt, target_actor(state_nxt))
        target_q_values = target_q_values * (1 - mask_nxt)
        td_target = reward + args.gamma * target_q_values

        critic_loss = F.mse_loss(critic(state_crt, actor(state_crt)), td_target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_loss1 = -torch.mean(critic(state_crt, actor(state_crt)))
        enc_criterion = torch.nn.BCELoss()
        action_nxt = action_nxt.to(torch.float)
        actor_loss2 = enc_criterion(actor(state_crt), action_nxt)
        actor_loss = args.epsilon * actor_loss1 + (1 - args.epsilon) * actor_loss2

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        actor_loss_list.append(actor_loss.data.cpu().numpy())
        critic_loss_list.append(critic_loss.data.cpu().numpy())

        soft_update(actor, target_actor)
        soft_update(critic, target_critic)

        action_nxt_prob = actor(state_crt)
        zero = torch.zeros_like(action_nxt_prob)
        one = torch.ones_like(action_nxt_prob)

        action_nxt_prob = torch.where(action_nxt_prob >= 0.5, one, action_nxt_prob)
        action_nxt_prob = torch.where(action_nxt_prob < 0.5, zero, action_nxt_prob)

        idx = torch.argmax(mask_nxt, 1)
        idx[idx == 0] = args.length
        shape = action_nxt_prob.shape[0]

        for b in range(shape):
            pred_action_nxt = action_nxt_prob.data.cpu()[b, :idx[b], :]
            action_nxt_lable = action_nxt.data.cpu()[b, :idx[b], :]
            j_score = utils.compute_jaccard(pred_action_nxt, action_nxt_lable)
            j_score_list.append(j_score)

            pred_action_nxt = pred_action_nxt.reshape(-1)
            action_nxt_lable = action_nxt_lable.reshape(-1)
            precision_list.append(precision_score(action_nxt_lable, pred_action_nxt, average='binary'))
            recall_list.append(recall_score(action_nxt_lable,pred_action_nxt,average='binary'))
            f1_list.append(f1_score(action_nxt_lable,pred_action_nxt,average='binary'))


    end = time.time()
    actor_loss_value = np.mean(actor_loss_list)
    critic_loss_value = np.mean(critic_loss_list)
    print('|train time {:0.5f}|'.format(end - start))

    jaccard = np.nanmean(j_score_list)
    precision = np.nanmean(precision_list)
    recall = np.nanmean(recall_list)
    f1 = np.nanmean(f1_list)
    f1_cal=2 * (precision * recall) / (precision + recall) 
    return actor_loss_value, critic_loss_value, precision, jaccard, recall,f1,f1_cal


def test_srl_run(epoch, actor, train_data, device):
    start = time.time()
    actor.eval()

    j_score_list = []
    precision_list=[]
    recall_list=[]
    f1_list=[]

    for data in train_data:
        state_crt, state_nxt, action_crt, action_nxt, mask_nxt, reward = data[:6]
        state_crt, state_nxt, action_crt, action_nxt, mask_nxt, reward = \
            state_crt.to(device), state_nxt.to(device), action_crt.to(device), action_nxt.to(device), mask_nxt.to(device), reward.to(device)

        action_nxt = action_nxt.to(torch.float)

        action_nxt_prob = actor(state_crt)
        zero = torch.zeros_like(action_nxt_prob)
        one = torch.ones_like(action_nxt_prob)
        prob = action_nxt_prob.clone()

        action_nxt_prob = torch.where(action_nxt_prob >= 0.4, one, action_nxt_prob)
        action_nxt_prob = torch.where(action_nxt_prob < 0.4, zero, action_nxt_prob)

        idx = torch.argmax(mask_nxt, 1)
        idx[idx == 0] = args.length
        shape = action_nxt_prob.shape[0]

        for b in range(shape):
            pred_action_nxt = action_nxt_prob.data.cpu()[b, :idx[b], :]
            action_nxt_lable = action_nxt.data.cpu()[b, :idx[b], :]
            j_score = utils.compute_jaccard(pred_action_nxt, action_nxt_lable)
            j_score_list.append(j_score)

            pred_action_nxt = pred_action_nxt.reshape(-1)
            action_nxt_lable = action_nxt_lable.reshape(-1)
            precision_list.append(precision_score(action_nxt_lable, pred_action_nxt, average='binary'))
            recall_list.append(recall_score(action_nxt_lable,pred_action_nxt,average='binary'))
            f1_list.append(f1_score(action_nxt_lable,pred_action_nxt,average='binary'))

    end = time.time()
    print('|test time {:0.5f}|'.format(end - start))
    jaccard = np.nanmean(j_score_list)

    precision = np.nanmean(precision_list)
    recall = np.nanmean(recall_list)
    f1 = np.nanmean(f1_list)
    f1_cal=2 * (precision * recall) / (precision + recall) 

    return precision, jaccard,  recall,f1,f1_cal

def reward_compute(actor, train_data, device):
    start = time.time()
    actor.eval()

    j_score_list = []
    reward_list = []

    for data in train_data:
        state_crt, state_nxt, action_crt, action_nxt, mask_nxt, reward = data[:6]
        state_crt, state_nxt, action_crt, action_nxt, mask_nxt, reward = \
            state_crt.to(device), state_nxt.to(device), action_crt.to(device), action_nxt.to(device), mask_nxt.to(device), reward.to(device)

        action_nxt = action_nxt.to(torch.float)

        ori_action_nxt_prob = actor(state_crt)
        zero = torch.zeros_like(ori_action_nxt_prob)
        one = torch.ones_like(ori_action_nxt_prob)

        ori_action_nxt_prob = torch.where(ori_action_nxt_prob >= 0.5, one, ori_action_nxt_prob)
        ori_action_nxt_prob = torch.where(ori_action_nxt_prob < 0.5, zero, ori_action_nxt_prob)

        idx = torch.argmax(mask_nxt, 1)
        idx[idx == 0] = args.length
        shape = ori_action_nxt_prob.shape[0]
        for b in range(shape):
            pred_action_nxt = ori_action_nxt_prob.data.cpu()[b, :idx[b], :]
            for day in range(0,idx[b]):
                drug_list=pred_action_nxt[day,:].view(-1)
                tmp_b=utils.reward_cal(drug_list,ddi_mat)
                reward[b][day]+=tmp_b
            reward_b=reward[b,:idx[b]]
            reward_list.append(torch.mean(reward_b).item())

            action_nxt_lable = action_nxt.data.cpu()[b, :idx[b], :]
            j_score = utils.compute_jaccard(pred_action_nxt, action_nxt_lable)
            j_score_list.append(j_score)
    
    end = time.time()
    print('|test time {:0.5f}|'.format(end - start))

    jaccard = np.nanmean(j_score_list)
    reward = np.mean(reward_list)
    print('ori jscore is ',jaccard)
    print('ori reward is ',reward)
    return jaccard, reward


def train_srl(train_data, test_data, device, epoch, hospitalid, per_model):
    actor = model.actor(input_size=args.state_size, hidden_size=args.hidden_size)
    if epoch == 0:
        if (os.path.exists(os.path.join(model_dir, 'model/actor_'+str(hospitalid)+'.pth'))):
            actor.load_state_dict(torch.load(os.path.join(model_dir, 'model/actor_'+str(hospitalid)+'.pth')))
            print('load successfully')
    actor = actor.to(device)

    critic = model.Critic(action_size=args.action_size, hidden_size=args.hidden_size)
    if epoch == 0:
        if (os.path.exists(os.path.join(model_dir, 'model/critic_'+str(hospitalid)+'.pth'))):
            critic.load_state_dict(torch.load(os.path.join(model_dir, 'model/critic_'+str(hospitalid)+'.pth')))
            print('load successfully')

    critic = critic.to(device)

    if type(per_model) != int:
        actor.load_state_dict(per_model)

    target_actor = model.actor(input_size=args.state_size, hidden_size=args.hidden_size)
    target_actor = target_actor.to(device)
    target_critic = model.Critic(action_size=args.action_size, hidden_size=args.hidden_size)
    target_critic = target_critic.to(device)
    target_critic.load_state_dict(critic.state_dict())
    target_actor.load_state_dict(actor.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    aloss_value, closs_value, p, j_value,r,f,f_score = train_srl_run(epoch, actor, critic, target_actor, target_critic,
                                                                 actor_optimizer, critic_optimizer, train_data, device)
    print('|Client {:d} on train set|jaccard: {:3.4f}|precision: {:3.4f}|r: {:3.4f}|f: {:3.4f}|'\
          .format(hospitalid,j_value,p,r,f))
    with open(os.path.join(model_dir, 'train_data_result' + str(hospitalid) + '.txt'), 'a', encoding='utf-8') as f:
        f.write('\t'.join([str(epoch), str(aloss_value), str(closs_value), str(p), str(j_value),str(r),str(f),str(f_score)]) + '\n')


    if epoch == (args.epoch_num - 1):
        test_acc_value, test_j_value, test_acc1, test_acc5, test_acc10 = test_srl_run(
            epoch, actor, test_data, device)
        print('|Client {:d} on test set|epoch: {:d}|jaccard: {:3.4f}|precision: {:3.4f}|acc1: {:3.4f}|acc5: {:3.4f}|acc10: {:3.4f}|'\
          .format(hospitalid, epoch, test_j_value,test_acc_value,test_acc1,test_acc5,test_acc10))
    
        with open(os.path.join(model_dir, 'new/test_data_result' + str(hospitalid) + '.txt'), 'a', encoding='utf-8') as f:
            f.write('\t'.join([str(epoch), str(test_acc_value), str(test_j_value), str(test_acc1), str(test_acc5), str(test_acc10)]) + '\n')

    torch.save(actor.state_dict(), os.path.join(model_dir, 'model/actor_'+str(hospitalid)+'.pth'))
    torch.save(critic.state_dict(), os.path.join(model_dir, 'model/critic_'+str(hospitalid)+'.pth'))

    actor = actor.cpu()
    model_weight = actor.state_dict()
    return model_weight


def main(hospital_list, device, epoch, CliOnSerDic, hospital_cluster):
    new_model_weights = []
    data_number_train=[]
    for hospital_id in hospital_list:
        print('---------hospital {:d} train---------'.format(hospital_id))
        icustayid_split_dict = utils.myreadjson(
            os.path.join(args.data_dir, 'feature/client/id' + str(hospital_id) + '_split_dict.json'))
        icustayid_train = icustayid_split_dict['icustayid_train']
        icustayid_test = icustayid_split_dict['icustayid_test']

        data_number=len(icustayid_train)

        start=time.time()
        dataset = data_loader.DataBowl(args, icustayid_train, hospital_id)
        train_data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)
        end=time.time()
        print('|load time {:0.5f}|'.format(end - start))

        if (epoch == args.epoch_num-1):
            start=time.time()
            dataset = data_loader.DataBowl(args, icustayid_test, hospital_id)
            test_data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)
            end=time.time()
            print('|load time {:0.5f}|'.format(end - start))

        if len(CliOnSerDic) == 0:
            per_model = 0
        else:
            per_model = CliOnSerDic[hospital_cluster[hospital_id]]

        if (epoch == args.epoch_num-1):
            client_weight = train_srl(train_data, test_data, device, epoch, hospital_id, per_model)
        else:
            client_weight = train_srl(train_data, train_data, device, epoch, hospital_id, per_model)

        new_model_weights.append(client_weight)
        data_number_train.append(data_number)
    return new_model_weights,data_number_train


def client_test(hospital_list, device, epoch, CliOnSerDic, usr_model_weights, hospital_cluster):
    reward_list=[]
    i=0
    for hospital_id in hospital_list:
        print('---------hospital {:d} calculate reward---------'.format(hospital_id))
        icustayid_split_dict = utils.myreadjson(
            os.path.join(args.data_dir, 'feature/client/id' + str(hospital_id) + '_split_dict.json'))
        icustayid_train = icustayid_split_dict['icustayid_train']

        start=time.time()
        dataset = data_loader.DataBowl(args, icustayid_train, hospital_id)
        test_data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)
        end=time.time()
        print('|load time {:0.5f}|'.format(end - start))

        ori_model = usr_model_weights[i]
        agg_model = CliOnSerDic[hospital_cluster[hospital_id]]
        
        ori_actor = model.actor(input_size=args.state_size, hidden_size=args.hidden_size)
        agg_actor = model.actor(input_size=args.state_size, hidden_size=args.hidden_size)
        ori_actor = ori_actor.to(device)
        agg_actor = agg_actor.to(device)

        ori_actor.load_state_dict(ori_model)
        agg_actor.load_state_dict(agg_model)

        ori_jscore, ori_reward = reward_compute(ori_actor,test_data, device)
        agg_jscore, agg_reward = reward_compute(agg_actor, test_data, device)
        print('jscore of client {:d} is {:0.5f}, {:0.5f}'.format(hospital_id,ori_jscore,agg_jscore))
        print('reward of client {:d} is {:0.5f}, {:0.5f}'.format(hospital_id,ori_reward,agg_reward))
        with open(model_dir+'policy_reward.txt','a',encoding='utf-8') as f:
            f.write('\t'.join([str(hospital_id),str(ori_jscore),str(agg_jscore),str(ori_reward),str(agg_reward)])+'\n')

        reward=args.server_alpha*(agg_jscore-ori_jscore)+(1-args.server_alpha)*(agg_reward-ori_reward)
        print('server reward is ',reward)
        reward_list.append(reward)
        i+=1

    return reward_list


if __name__ == '__main__':
    gpu_index = [1,2,3,4,5]
    num_processes = len(gpu_index)
    devices = [torch.device('cuda', index) for index in gpu_index]
    hospitalid = np.load('../../data/feature/client/hospitalid.npy')
    hospitalid = hospitalid[:]

    tmp_hospitalid=list()
    for i in range(num_processes):
        j=list(hospitalid[range(i,len(hospitalid),num_processes)])
        tmp_hospitalid.extend(j)
    hospitalid=tmp_hospitalid

    step = int(np.ceil(len(hospitalid) / num_processes))
    per_fed_model = 0
    temper = 1.0
    cluster_num = args.cluster_num
    CliOnSerDic = {}
    chosenCliOnSerDic = {}
    perFedModel4Cli = {}
    perFedIndex4Cli = []
    hospital_cluster = {}

    policy_network = model.PersonalFederatedModel(len(hospitalid), cluster_num, temper, 'Avg')
    if (os.path.exists(os.path.join(model_dir, 'model/policy_network.pth'))):
        policy_network.load_state_dict(torch.load(os.path.join(model_dir, 'model/policy_network.pth')))
        print('load successfully')
    PN_optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.server_learning_rate)

    print('---------Start training---------')
    for epoch in range(args.epoch_num):
        time_start = time.time()
        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=num_processes)
        process_arr = []
        for i in range(num_processes):
            device = devices[i]
            process_arr.append(
                pool.apply_async(main, args=(
                hospitalid[i * step:(i + 1) * step], devices[i], epoch, CliOnSerDic, hospital_cluster)))
        pool.close()
        pool.join()
        print('---------Finishing training clients---------')
        usr_model_weights_t = []
        usr_data_number_t=[]
        if num_processes > 1:
            usr_model_weights_t,usr_data_number = process_arr[0].get()
            for process in process_arr[1:]:
                tmp_usr_model_weights_t,tmp_usr_data_number=process.get()
                usr_model_weights_t+=tmp_usr_model_weights_t
                usr_data_number_t+=tmp_usr_data_number
            usr_model_weights=usr_model_weights_t
            usr_data_number=usr_data_number_t
        else:
            usr_model_weights,usr_data_number = process_arr[0].get()

        time_end = time.time()
        time_c = time_end - time_start
        print('---------Time cost of train of epoch', epoch, ':', time_c, 's---------')
        print('user model weights is :', len(usr_model_weights))
        print('user data number is ', usr_data_number)

        policy_prob,CliOnSerDic, chosenCliOnSerDic = policy_network(usr_model_weights, {}, 1.2)
        print('cluster result is ',chosenCliOnSerDic)
        with open(os.path.join(model_dir, 'cluster_result.txt'), 'a', encoding='utf-8') as f:
            f.write('\t'.join([str(chosenCliOnSerDic)]) + '\n')

        perFedModel4Cli = {}
        for per_fed_ind in chosenCliOnSerDic:
            for local_model_ind in chosenCliOnSerDic[per_fed_ind]:
                perFedModel4Cli[local_model_ind] = per_fed_ind
        perFedIndex4Cli = []
        for indd in range(len(perFedModel4Cli.keys())):
            perFedIndex4Cli.append(perFedModel4Cli[indd])

        for perFedKey in CliOnSerDic:
            torch.save(CliOnSerDic[perFedKey], os.path.join(model_dir, 'model/cluster_'+str(perFedKey)+'.pth'))
            for key in CliOnSerDic[perFedKey]:
                CliOnSerDic[perFedKey][key] = CliOnSerDic[perFedKey][key].detach()

        hospital_cluster = {}
        for i in range(len(hospitalid)):
            hospital_cluster[hospitalid[i]]=perFedIndex4Cli[i]
        print('hospital_cluster is ', hospital_cluster)
        print('---------Finishing fisrt aggregation---------')

        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=num_processes)
        process_arr = []
        for i in range(num_processes):
            device = devices[i]
            process_arr.append(
                pool.apply_async(client_test, args=(
                    hospitalid[i * step:(i + 1) * step], devices[i], epoch, CliOnSerDic, usr_model_weights[i * step:(i + 1) * step], hospital_cluster)))
        pool.close()
        pool.join()
        print('---------Finishing client test---------')

        user_reward_t = []
        if num_processes > 1:
            user_reward_t = process_arr[0].get()
            for process in process_arr[1:]:
                tmp_user_reward_t=process.get()
                user_reward_t+=tmp_user_reward_t
            user_reward = user_reward_t
        else:
            user_reward = process_arr[0].get()

        print('user_reward',user_reward)
        with open(model_dir+'policy_reward.txt','a',encoding='utf-8') as f:
            f.write(str(user_reward)+'\n')

        policy_reward=np.nanmean(user_reward)
        print('policy_reward',policy_reward)
        action_prob_sum=(policy_prob[:,perFedIndex4Cli]).sum()
        grad_term=policy_reward*action_prob_sum
        print('grad_term',grad_term)
        PN_optimizer.zero_grad()
        grad_term.backward()
        PN_optimizer.step()
        print('---------Policy Network has been updated---------')
        with open(model_dir+'policy_reward.txt','a',encoding='utf-8') as f:
            f.write(str(policy_reward)+'\n')
        torch.save(policy_network.state_dict(), os.path.join(model_dir, 'model/policy_network.pth'))
        # ------------------------------------------------------------------------------------------------
        time_last = time.time()
        time_avg = time_last - time_start
        print('***************Time cost of Fedavg of epoch', epoch, ':', time_avg, '***************')
