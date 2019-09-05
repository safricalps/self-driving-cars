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
        super(BPNetWork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.connection1 = nn.Linear(input_size, 30)
        self.connection2 = nn.Linear(30, output_size)
    
    def forward(self, environment_data):
        hiddenLayers = fun.relu(self.connection1(environment_data))
        qValues = self.connection2(hiddenLayers)
        return qValues

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, train_size):
        samples = zip(*random.sample(self.memory, train_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

class DeepQNetwork():
    def __init__(self, input_size, output_size, gamma):
        self.gamma = gamma
        #self.reward_average = []
        self.model = BPNetWork(input_size, output_size)
        self.memory = ExperienceReplay(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_data = torch.Tensor(np.zeros(15)).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, environment_data):
        probs = fun.softmax(self.model.forward(Variable(environment_data, volatile=True)) * 100)
        # print(probs)
        action = torch.multinomial(probs.squeeze(1),5,replacement=True)
        return action.data[0,0]

    def learn(self, data_one, data_second, reward, action, status):
        q_estimate = self.model.forward(data_one).gather(1, action.unsqueeze(1)).squeeze(1)
        output = self.model.forward(data_second).detach().max(1)[0]
        q_target = np.zeros(100)
        q_target = self.gamma * output + reward
        #q_target[status == 1] = reward[status == 1]
        td_loss = fun.smooth_l1_loss(q_estimate, q_target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def update(self, reward, new_data, status):
        new_data = torch.Tensor(new_data).float().unsqueeze(0)
        action = self.select_action(new_data)

        self.memory.push((self.last_data, new_data, torch.Tensor([reward]), torch.LongTensor([int(self.last_action)]), torch.Tensor([status])))
        
        if len(self.memory.memory) > 100:
            data_one, data_second, reward, action, status = self.memory.sample(100)
            self.learn(data_one, data_second, reward, action, status)
            
        self.last_action = action
        self.last_data = new_data
        #self.last_reward = reward
        #self.reward_average.append(reward)
        #if len(self.reward_average) > 1000:
            #del self.reward_average[0]
        return action

    #def score(self):
        #return sum(self.reward_average) / (len(self.reward_average) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'saved_ai.pth')

    def load(self):
        if os.path.isfile('saved_ai.pth'):
            print("Loading checkpoint...")
            checkpoint = torch.load('saved_ai.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("No checkpoint found...")
