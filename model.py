import sys

import os
import sys
import time
import random
import json
from collections import OrderedDict
from tqdm import tqdm
from torch.optim import Optimizer


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time

import parse
import numpy as np
from sklearn import metrics
import copy

class actor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(actor, self).__init__()

        self.encoder = nn.LSTM(239, hidden_size, dropout=0.2, batch_first=True, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(4*hidden_size, 306)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        state_nxt, _ = self.encoder(state)
        nxt = self.fc(state_nxt)
        prob = self.sigmoid(nxt)
        return prob


class Critic(nn.Module):
    def __init__(self, action_size, hidden_size):
        super(Critic, self).__init__()
        self.encoder = nn.LSTM(239, hidden_size, batch_first=True, num_layers=3)  # input:(batch_size,seq_len,inputsize)

        self.fc_nxt = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size),
            nn.Dropout(p=0.3),
            # nn.PReLU(),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(p=0.3),
            # nn.PReLU(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action):
        state_nxt, _ = self.encoder(state)
        input = torch.concat([state_nxt,action],dim=2)
        output_nxt = self.fc_nxt(input)
        output_nxt = self.sigmoid(output_nxt)
        output = torch.squeeze(output_nxt)
        return output


def getGumbelSoftmax(pro, temper, scalar):
    return nn.functional.gumbel_softmax(nn.functional.softmax(pro, 1) * scalar, temper, False)


def getInitValForGS(batch_size, num, mode):
    if mode == 'Random':
        return torch.tensor(np.random.random((batch_size, num)))
    elif mode == 'Avg':
        return torch.tensor(20 * np.ones((batch_size, num)))


def updateWeights(weights, modelWeights):
    weight_res = copy.deepcopy(modelWeights[0])
    for k in weight_res.keys():
        weight_res[k] = weight_res[k] * weights[0]
        for i in range(1, len(modelWeights)):
            weight_res[k] += modelWeights[i][k] * weights[i]
    return weight_res


def PerFedUpdateWeight(outputOfPerFedModel, modelStaLis, exceptLayer):
    maxpos = outputOfPerFedModel.max(1).indices
    numOfCliOnSer = int(maxpos.max()) + 1
    CliOnSerDic = {}
    for ind in range(len(maxpos)):
        if int(maxpos[ind]) in CliOnSerDic.keys():
            CliOnSerDic[int(maxpos[ind])].append(ind)
        else:
            CliOnSerDic[int(maxpos[ind])] = [ind]
    weightsOfCliOnSer = []
    for i in range(numOfCliOnSer):
        weights = outputOfPerFedModel[:, i] / sum(outputOfPerFedModel[:, i])
        weightsOfCliOnSer.append(updateWeights(weights, modelStaLis))
    return weightsOfCliOnSer, CliOnSerDic


def PerFedUpdateWeightGroup(outputOfPerFedModel, modelStaLis, LastCliOnSerDic={}):
    maxpos = outputOfPerFedModel.max(1).indices
    numOfCliOnSer = outputOfPerFedModel.shape[1]
    CliOnSerDic = {}
    for ind in range(len(maxpos)):
        if int(maxpos[ind]) in CliOnSerDic.keys():
            CliOnSerDic[int(maxpos[ind])].append(ind)
        else:
            CliOnSerDic[int(maxpos[ind])] = [ind]
    for ind in range(numOfCliOnSer):
        if ind not in CliOnSerDic:
            CliOnSerDic[ind] = []
    CliOnSerDic = dict([(k, CliOnSerDic[k]) for k in sorted(CliOnSerDic.keys())])
    weightsOfCliOnSer = {}
    for i in range(numOfCliOnSer):
        if i in CliOnSerDic.keys():
            if len(CliOnSerDic[i]) != 0:
                weights = outputOfPerFedModel[CliOnSerDic[i], i] / sum(outputOfPerFedModel[CliOnSerDic[i], i])
                tem_modelStaLis = []
                for ind in CliOnSerDic[i]:
                    tem_modelStaLis.append(modelStaLis[ind])
                weightsOfCliOnSer[i] = updateWeights(weights, tem_modelStaLis)
    return weightsOfCliOnSer, CliOnSerDic


class PersonalFederatedModel(nn.Module):
    def __init__(self, batchSize, numOfCliOnSer, temper, initalMode):
        super().__init__()
        self.numOfCliOnSer = numOfCliOnSer
        self.probability = nn.Parameter(getInitValForGS(batchSize, numOfCliOnSer, initalMode), requires_grad=True)
        self.GumbelSoftmax = getGumbelSoftmax
        self.temper = temper
        self.PerFedUpdateWeight = PerFedUpdateWeightGroup

    def forward(self, stateLis, LastCliOnSerDic, scalar):
        raw_pro = self.probability
        result_gs = self.GumbelSoftmax(raw_pro, self.temper, scalar)
        CliOnSerDic, chosenCliOnSerDic = self.PerFedUpdateWeight(result_gs, stateLis, LastCliOnSerDic)
        return result_gs,CliOnSerDic, chosenCliOnSerDic

