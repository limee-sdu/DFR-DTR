import os
import json
import time
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import sys


class DataBowl(Dataset):
    def __init__(self, args, icu_id,hospital_id):
        self.args = args
        self.icustayid_list = icu_id

        lab_state = pd.read_csv(os.path.join(args.data_dir, 'feature/client/lab_'+str(hospital_id)+'.csv'))
        diagnose = pd.read_csv(os.path.join(args.data_dir, 'feature/client/diagnosis_'+str(hospital_id)+'.csv'))
        action = pd.read_csv(os.path.join(args.data_dir, 'feature/client/action_'+str(hospital_id)+'.csv'))
        patient = pd.read_csv(os.path.join(args.data_dir, 'feature/client/patient_'+str(hospital_id)+'.csv'))
        trans_reward = pd.read_csv(os.path.join(args.data_dir, 'feature/client/lab_reward_'+str(hospital_id)+'.csv'))
        # drug_reward = pd.read_csv(os.path.join(args.data_dir, 'feature/client/drug_reward_'+str(hospital_id)+'.csv'))

        self.state = lab_state
        self.diagnose = diagnose
        self.action = action
        self.patient = patient
        self.trans_reward = trans_reward
        # self.drug_reward = drug_reward

    def __getitem__(self, idx, split=None):
        patientunitstayid = self.icustayid_list[idx]
        lab_state = self.state[self.state['patientunitstayid'] == patientunitstayid].values
        lab_state = lab_state[:-1, 1:]
        # lab_state=[time_len, 133]
        diagnose_state = self.diagnose[self.diagnose['patientunitstayid'] == patientunitstayid].values
        diagnose_state = diagnose_state[:, 1:]
        # diagnose_state : [1,100]
        sta_state = self.patient[self.patient['patientunitstayid'] == patientunitstayid].values
        sta_state = sta_state[:, 1:-1]
        # sta_state : [1,6]
        action = self.action[self.action['patientunitstayid'] == patientunitstayid].values
        action = action[:, 2:]
        # action : [time_len, 306]
        # action = np.array(action).transpose(1, 0)
        mortality = self.patient[self.patient['patientunitstayid'] == patientunitstayid]['hospitaldischargestatus']. \
            values.astype(np.float32)
        # mortality 0 表示存活 1 表示死亡
        trans_reward = self.trans_reward[self.trans_reward['patientunitstayid'] == patientunitstayid].values
        trans_reward = trans_reward[:, -1]

        # drug_reward = self.drug_reward[self.drug_reward['patientunitstayid'] == patientunitstayid].values
        # drug_reward = drug_reward[:, -1]

        size = lab_state.shape

        # 合并 动态state和静态state
        diagnose_state = np.repeat(diagnose_state, size[0], axis=0)
        # print(diagnose_state.shape)
        lab_state = np.concatenate((lab_state, diagnose_state), 1)
        sta_state = np.repeat(sta_state, size[0], axis=0)
        lab_state = np.concatenate((lab_state, sta_state), 1)

        n = self.args.length
        delta = 1
        size = lab_state.shape

        if size[0] >= n + delta:
            # 只取前n个state
            state = lab_state[-n - delta:]
            action = action[-n - delta:]
            trans_reward = trans_reward[-n:]
            # drug_reward = drug_reward[-n:]
            reward = np.zeros(n, dtype=np.float32)
            reward[-1] = (0.5 - mortality) * 2 * 15
            reward = np.sum([reward, trans_reward], axis=0).tolist()
            mask = np.zeros(n + delta, dtype=int)
        else:
            # 补齐
            padding = np.zeros((n + delta - size[0], size[1]))
            state = np.concatenate((lab_state, padding), 0)
            size2 = action.shape
            padding = np.zeros((n + delta - size[0], size2[1]))
            action = np.concatenate((action, padding), 0)
            mask = np.zeros(n + delta, dtype=int)
            mask[size[0]:] = 1  # padding部分为1
            reward = np.zeros(n, dtype=np.float32)
            reward[size[0] - 1] = (0.5 - mortality) * 2 * 15
            padding = np.zeros(n - trans_reward.size, dtype=np.float32)
            trans_reward = np.concatenate((trans_reward, padding), 0)
        
            reward = np.sum([reward, trans_reward], axis=0).tolist()

        state = state.astype(np.float32)
        mask = mask.astype(np.float32)
        action = action.astype(np.int64)

        # state = np.array(state).transpose(1, 0)
        # mask = np.array(mask).transpose(1, 0)
        # action = np.array(action).transpose(1, 0)

        state_crt = state[: -delta]
        state_nxt = state[delta:]
        action_crt = action[: -delta]
        action_nxt = action[delta:]
        # mask_crt = mask[: -delta]
        mask_nxt = mask[delta:]
        # if action_crt.shape[0]>300:
        #     print("ERROR")

        return torch.from_numpy(state_crt), torch.from_numpy(state_nxt), torch.from_numpy(action_crt), \
               torch.from_numpy(action_nxt), torch.from_numpy(mask_nxt), torch.Tensor(reward), patientunitstayid

    def __len__(self):
        return len(self.icustayid_list)


class noreward_DataBowl(Dataset):
    def __init__(self, args, icu_id,hospital_id):
        self.args = args
        self.icustayid_list = icu_id

        lab_state = pd.read_csv(os.path.join(args.data_dir, 'feature/client/lab_'+str(hospital_id)+'.csv'))
        diagnose = pd.read_csv(os.path.join(args.data_dir, 'feature/client/diagnosis_'+str(hospital_id)+'.csv'))
        action = pd.read_csv(os.path.join(args.data_dir, 'feature/client/action_'+str(hospital_id)+'.csv'))
        patient = pd.read_csv(os.path.join(args.data_dir, 'feature/client/patient_'+str(hospital_id)+'.csv'))
        # trans_reward = pd.read_csv(os.path.join(args.data_dir, 'feature/trans_reward420_cal.csv'))
        # drug_reward = pd.read_csv(os.path.join(args.data_dir, 'feature/drug_reward420.csv'))

        self.state = lab_state
        self.diagnose = diagnose
        self.action = action
        self.patient = patient
        # self.trans_reward = trans_reward
        # self.drug_reward = drug_reward

    def __getitem__(self, idx, split=None):
        patientunitstayid = self.icustayid_list[idx]
        lab_state = self.state[self.state['patientunitstayid'] == patientunitstayid].values
        lab_state = lab_state[:-1, 1:]
        # lab_state=[time_len, 133]
        diagnose_state = self.diagnose[self.diagnose['patientunitstayid'] == patientunitstayid].values
        diagnose_state = diagnose_state[:, 1:]
        # diagnose_state : [1,100]
        sta_state = self.patient[self.patient['patientunitstayid'] == patientunitstayid].values
        sta_state = sta_state[:, 1:-1]
        # sta_state : [1,6]
        action = self.action[self.action['patientunitstayid'] == patientunitstayid].values
        action = action[:, 2:]
        # action : [time_len, 306]
        # action = np.array(action).transpose(1, 0)
        mortality = self.patient[self.patient['patientunitstayid'] == patientunitstayid]['hospitaldischargestatus']. \
            values.astype(np.float32)
        # mortality 0 表示存活 1 表示死亡
        # trans_reward = self.trans_reward[self.trans_reward['patientunitstayid'] == patientunitstayid].values
        # trans_reward = trans_reward[:, -1]
        #
        # drug_reward = self.drug_reward[self.drug_reward['patientunitstayid'] == patientunitstayid].values
        # drug_reward = drug_reward[:, -1]

        size = lab_state.shape

        # 合并 动态state和静态state
        diagnose_state = np.repeat(diagnose_state, size[0], axis=0)
        # print(diagnose_state.shape)
        lab_state = np.concatenate((lab_state, diagnose_state), 1)
        sta_state = np.repeat(sta_state, size[0], axis=0)
        lab_state = np.concatenate((lab_state, sta_state), 1)

        n = self.args.length
        delta = 1
        size = lab_state.shape

        if size[0] >= n + delta:
            # 只取前n个state
            state = lab_state[-n - delta:]
            action = action[-n - delta:]
            # trans_reward = trans_reward[-n:]
            # drug_reward = drug_reward[-n:]
            reward = np.zeros(n, dtype=np.float32)
            reward[-1] = (0.5 - mortality) * 2 * 15
            # reward = np.sum([reward, trans_reward, drug_reward], axis=0).tolist()
            mask = np.zeros(n + delta, dtype=int)
        else:
            # 补齐
            padding = np.zeros((n + delta - size[0], size[1]))
            state = np.concatenate((lab_state, padding), 0)
            size2 = action.shape
            padding = np.zeros((n + delta - size[0], size2[1]))
            action = np.concatenate((action, padding), 0)
            mask = np.zeros(n + delta, dtype=int)
            mask[size[0]:] = 1  # padding部分为1
            reward = np.zeros(n, dtype=np.float32)
            reward[size[0] - 1] = (0.5 - mortality) * 2 * 15
            # padding = np.zeros(n - trans_reward.size, dtype=np.float32)
            # trans_reward = np.concatenate((trans_reward, padding), 0)
            # padding = np.zeros(n - drug_reward.size, dtype=np.float32)
            # drug_reward = np.concatenate((drug_reward, padding), 0)
            # reward = np.sum([reward, trans_reward, drug_reward], axis=0).tolist()

        state = state.astype(np.float32)
        mask = mask.astype(np.float32)
        action = action.astype(np.int64)

        # state = np.array(state).transpose(1, 0)
        # mask = np.array(mask).transpose(1, 0)
        # action = np.array(action).transpose(1, 0)

        state_crt = state[: -delta]
        state_nxt = state[delta:]
        action_crt = action[: -delta]
        action_nxt = action[delta:]
        # mask_crt = mask[: -delta]
        mask_nxt = mask[delta:]
        # if action_crt.shape[0]>300:
        #     print("ERROR")

        return torch.from_numpy(state_crt), torch.from_numpy(state_nxt), torch.from_numpy(action_crt), \
               torch.from_numpy(action_nxt), torch.from_numpy(mask_nxt), torch.Tensor(reward), patientunitstayid

    def __len__(self):
        return len(self.icustayid_list)


class LG_DataBowl(Dataset):
    def __init__(self, args, icu_id,hospital_id):
        self.args = args
        self.icustayid_list = icu_id

        lab_state = pd.read_csv(os.path.join(args.data_dir, 'feature/client/lab_'+str(hospital_id)+'.csv'))
        diagnose = pd.read_csv(os.path.join(args.data_dir, 'feature/client/diagnosis_'+str(hospital_id)+'.csv'))
        action = pd.read_csv(os.path.join(args.data_dir, 'feature/client/action_'+str(hospital_id)+'.csv'))

        self.state = lab_state
        self.diagnose = diagnose
        self.action = action

    def __getitem__(self, idx, split=None):
        patientunitstayid = self.icustayid_list[idx]
        lab_state = self.state[self.state['patientunitstayid'] == patientunitstayid].values
        lab_state = lab_state[:-1, 1:]
        # lab_state=[time_len, 133]
        diagnose_state = self.diagnose[self.diagnose['patientunitstayid'] == patientunitstayid].values
        diagnose_state = diagnose_state[:, 1:]
        # diagnose_state : [1,100]
        action = self.action[self.action['patientunitstayid'] == patientunitstayid].values
        action = action[:, 2:]
        # action : [time_len, 306]
        # action = np.array(action).transpose(1, 0)

        size = lab_state.shape

        # 合并 动态state和静态state
        diagnose_state = np.repeat(diagnose_state, size[0], axis=0)
        # print(diagnose_state.shape)
        lab_state = np.concatenate((lab_state, diagnose_state), 1)

        n = self.args.length
        delta = 1
        size = lab_state.shape

        if size[0] >= n + delta:
            # 只取前n个state
            state = lab_state[-n - delta:]
            action = action[-n - delta:]
            mask = np.zeros(n + delta, dtype=int)
        else:
            # 补齐
            padding = np.zeros((n + delta - size[0], size[1]))
            state = np.concatenate((lab_state, padding), 0)
            size2 = action.shape
            padding = np.zeros((n + delta - size[0], size2[1]))
            action = np.concatenate((action, padding), 0)
            mask = np.zeros(n + delta, dtype=int)
            mask[size[0]:] = 1  # padding部分为1

        state = state.astype(np.float32)
        mask = mask.astype(np.float32)
        action = action.astype(np.int64)

        state_crt = state[: -delta]
        state_nxt = state[delta:]
        action_crt = action[: -delta]
        action_nxt = action[delta:]
        # mask_crt = mask[: -delta]
        mask_nxt = mask[delta:]

        return torch.from_numpy(state_crt), torch.from_numpy(state_nxt), torch.from_numpy(action_crt), \
               torch.from_numpy(action_nxt), torch.from_numpy(mask_nxt), patientunitstayid

    def __len__(self):
        return len(self.icustayid_list)


class rl_dataBowl(Dataset):
    def __init__(self, args, icu_id, hospital_id):
        self.args = args
        self.icustayid_list = icu_id
        
        done = pd.read_csv(os.path.join(args.data_dir, 'feature/done420.csv')) 

        lab_state = pd.read_csv(os.path.join(args.data_dir, 'feature/client/lab_'+str(hospital_id)+'.csv'))
        diagnose = pd.read_csv(os.path.join(args.data_dir, 'feature/client/diagnosis_'+str(hospital_id)+'.csv'))
        action = pd.read_csv(os.path.join(args.data_dir, 'feature/client/action_'+str(hospital_id)+'.csv'))
        patient = pd.read_csv(os.path.join(args.data_dir, 'feature/client/patient_'+str(hospital_id)+'.csv'))


        self.state = lab_state
        self.diagnose = diagnose
        self.action = action
        self.patient = patient
        self.done = done

    def __getitem__(self, idx, split=None):
        patientid=self.done.at[idx,'patientunitstayid']
        time = self.done.at[idx,'labresultoffset']

        lab_state = self.state.loc[idx,:].values
        lab_state = lab_state[2:]
        # lab_state=[132]
        diagnose_state = self.diagnose[self.diagnose['patientunitstayid'] == patientid].values
        diagnose_state = diagnose_state[:,1:]
        diagnose_state = diagnose_state.reshape(-1)
        # diagnose_state : [100]
        sta_state = self.patient[self.patient['patientunitstayid'] == patientid].values
        sta_state = sta_state[:,1:-1]
        sta_state = sta_state.reshape(-1)
        # sta_state : [6]
        action = self.action[self.action['patientunitstayid'] == patientid]
        action = action[action['time']==time].values
        action = action[:,2:]
        if action.shape[0]==0:
            action = np.zeros(306, dtype=np.float32)
        action = action.reshape(-1)
        action = action[:306]
        # action : [306]
        # action = np.array(action).transpose(1, 0)
        mortality = self.patient[self.patient['patientunitstayid'] == patientid]['hospitaldischargestatus'].values.astype(np.float32)
        # mortality 0 表示存活 1 表示死亡
        done=self.done.loc[idx,:].values.astype(np.float32)
        done=done[-1:]
        n=lab_state.shape

        if done==1:
            next_state= np.zeros(n, dtype=np.float32)
            if mortality:
                reward = -15
            else:
                reward = 15
        else:
            next_state=self.state.loc[idx+1,:].values
            next_state = next_state[2:]
            reward = 0

        # 合并 动态state和静态state
        sta_state=np.concatenate((sta_state,diagnose_state),0)

        # print(diagnose_state.shape)
        state_crt = np.concatenate((lab_state, sta_state), 0)
        state_nxt = np.concatenate((next_state, sta_state), 0)
        # [1,238]
        reward = np.array([reward])
        state_crt = state_crt.astype(np.float32)
        state_nxt = state_nxt.astype(np.float32)
        action = action.astype(np.int64)
        reward = reward.astype(np.int64)

        return torch.from_numpy(state_crt), torch.from_numpy(state_nxt), torch.from_numpy(action), \
               torch.from_numpy(mortality), torch.from_numpy(done), torch.from_numpy(reward)

    def __len__(self):
        return len(self.done)
