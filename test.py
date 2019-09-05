#-*- coding=utf-8 -&-

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class BPNetWork(nn.Module):
    def __init__(self, input_size, output_size):
        super(BPNetWork, self ).__init__() 
        self.input_size = input_size
        self.output_size = output_size
        self.connection1 = nn.Linear(input_size, 30)
        self.connection2 = nn.Linear(30, output_size)
    
    def forward(self, environment_data):
        hiddenLayers = fun.relu(self.connection1(environment_data))
        qValues = self.connection2(hiddenLayers)
        return qValues

model = BPNetWork(5,5)
action = torch.LongTensor([3])
status = torch.Tensor([1,0,1])
d = torch.Tensor([[0,0,0,0,0],[1,1,1,1,1],[2,2,2,2,2]]).float()
o = model.forward(d).detach().max(1)[0]
print(o[status == 1])
'''
memory = []
for i in range(5):
    memory.append((torch.Tensor([0,0,0,0,0]).unsqueeze(0),torch.Tensor([0,0,0,0,0]).unsqueeze(0),torch.Tensor([10]), torch.LongTensor([int(0)])))
a,b,c,d = map(lambda x: Variable(torch.cat(x, 0)),zip(*memory))
print(a.shape)
q_estimate = model.forward(a).gather(1, d.unsqueeze(1)).squeeze(1)
output = model.forward(b).detach().max(1)[0]
print(q_estimate)
print(model.forward(torch.Tensor([0,0,0,0,0]).gather(1, action.unsqueeze(1)).squeeze(1)))
'''