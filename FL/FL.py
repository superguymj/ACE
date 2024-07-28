import numpy as np
import random
import time
import os

import argparse
import torch
import copy

from collections import OrderedDict
from model.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from torch.utils.data import DataLoader
from data import Dataset_CIFAR10
from sam import SAM


eps = 1e-30
inf = 1e30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Client(object):
    pi = 0.5                # the number of CPU Kcycles to process a single data sample, Kcycles/bit
    C = 67 * 1024**2 * 8    # the data size of the uploaded model parameters in bits
    N0 = 1e-13              # the background noise
    epochs = 1
    batchSize = 128

    def __init__(self,
                 id,
                 freq,      # the CPU operating frequency interval, Hz
                 B,         # the resource blocks, Hz
                 p,         # the transmission power, W
                 h,         # the channel gain
                 Xi,
                 Battery,
                 bfi,       # the Big-Five traits
                 sat,       # the satisfaction of occupation-reward levels
                 model,     # the satisfaction model
                 pred,
                 true,
                 trainData,
                 testData
                 ):
        self.id = id
        self.Xi = Xi
        self.freq = freq
        
        c, w, h = trainData[0][0].shape
        self.D = (len(trainData) + len(testData)) * c * w * h * 128
        self.Tcomm = Client.C / (B * np.log2(1 + p * h * h / Client.N0))
        self.Ecomm = p * self.Tcomm
        self.Battery = Battery

        self.trainData = DataLoader(trainData, batch_size=Client.batchSize, shuffle=True, num_workers=2)
        self.testData = DataLoader(testData, batch_size=Client.batchSize, shuffle=False, num_workers=2)

        w0, w1 = bfi.weight(model)
        eta1, eta2 = model.eta1 * sat.scaler.std + sat.scaler.mean, model.eta2 * sat.scaler.std
        self.sens = eta2 * (1 - w0 - w1)
        w0 *= eta2
        w1 *= eta2
        # the satisfaction is equal to (eta + w0 * (1 - idle) + w1 * (1 - train) + sens * reward)
        self.w0, self.w1, self.eta = w0, w1, eta1

        self.pred = []; self.true = []
        for (i, j) in zip(pred, true):
            self.pred.append(max(min(100, i[0][0]), 0) / 100)
            self.true.append(max(min(100, j[0][0]), 0) / 100)
        
        self.pred = self.pred + self.true

        self.shortest = self.Ttrain(freq[1] - freq[0]) + self.Tcomm

        self.inv1 = 1 / (self.freq[1] - self.freq[0])
        self.inv2 = 1 / self.sens
        
        # print(self.Battery / self.Euser(freq[1], 1))
        # print(self.shortest)
        print(self.Battery)

    def Ttrain(self, freq):
        return Client.pi * self.D / freq

    def Etrain(self, freq):
        return self.Xi * self.D * Client.pi * freq * freq

    def Euser(self, freq, T):
        return self.Xi * freq * freq * freq * T

    def frequent(self, prec):
        return (self.freq[1] - self.freq[0]) * prec

    def percentage(self, freq):
        return freq * self.inv1

    def Train(self, Deadline):
        if Deadline < self.shortest:
            return None, None
        freq = Client.pi * self.D / (Deadline - self.Tcomm)
        return self.percentage(freq), freq

    def Reward(self, idle, train, Limit_min):
        if train is None:
            return None
        return max(0, (Limit_min - self.eta - self.w0 * (1 - idle) - self.w1 * (1 - train)) * self.inv2)

    def Satisfaction(self, idle, train, reward):
        if reward is None:
            return None
        return self.eta + self.w0 * (1 - idle) + self.w1 * (1 - train) + self.sens * reward

    def Energy(self, idle, train, T):
        idle = self.frequent(idle) + self.freq[0]
        if train is None:
            return self.Euser(idle, T), None
        return self.Euser(idle, T), self.Ecomm + self.Euser(idle, self.Tcomm) + self.Euser(min(self.freq[1], idle + train), T - self.Tcomm)
    
    def train(self, model, k, K):
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = StepLR(optimizer, 0.1, K); scheduler(k)
        
        for epoch in range(Client.epochs):
            model.train()
            for batch in self.trainData:
                inputs, targets = (b.to(device) for b in batch)

                # first forward-backward step
                enable_running_stats(model)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=0.1)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=0.1).mean().backward()
                optimizer.second_step(zero_grad=True)

                with torch.no_grad():
                    correct = torch.argmax(predictions.data, 1) == targets

            model.eval()

            with torch.no_grad():
                acc = 0
                total = 0
                for batch in self.testData:
                    inputs, targets = (b.to(device) for b in batch)

                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    acc += correct.cpu().sum().item()
                    total += loss.size(0)
                    
                print('\tClient{:02d} - Epoch{:02d} acc is {}'.format(self.id, epoch, acc / total))
        
        return model


from model.wide_res_net import WideResNet
from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

class Server(object):
    K = 192                     # the training round
    alpha = 0.95                # the decay factor
    Model = WideResNet          # the neural networks model
    Channels = 3                # the channels of picture
    Labels = 10                 # the labels of picture

    def __init__(self, clients, Deadline, Limit_avg, testDatas):
        self.n = len(clients)
        self.Budget = self.n * Server.K / 2
        self.clients = clients
        self.Deadline = Deadline
        self.Limit_avg = Limit_avg
        
        Server.Labels = len(testDatas.classes)
        self.initModel()
        
        self.test = DataLoader(testDatas, batch_size=128, shuffle=False, num_workers=2)
 
    def initModel(self):
        self.model = Server.Model(labels=Server.Labels).to(device)

    def clientSelection(self):
        pass
    
    def FedAvg(self, models):
        if len(models) == 0: 
            return
        total_weight = 0.
        base = OrderedDict()
        for (client_model, client_samples) in models:
            total_weight += client_samples
            for key, value in client_model.state_dict().items():
                # print(key)
                if key in base:
                    base[key] += (client_samples * value)
                else:
                    base[key] = (client_samples * value)

        averaged_soln = copy.deepcopy(self.model.state_dict())
        for key, value in base.items():
            if total_weight != 0:
                averaged_soln[key] = value.to(device) / total_weight
                
        self.model.load_state_dict(averaged_soln)

    def train(self, selection, setting):
        path = './results/{}'.format(setting)
        accs = []
        
        for k in range(Server.K):
            print('The {}-th round:'.format(k))
            models = []
            for client in self.clients: 
                if selection[k][client.id] == True:
                    model = copy.deepcopy(self.model).to(device)
                    models.append((client.train(model, k, Server.K), client.D))
            self.FedAvg(models)

            self.model.eval()
            with torch.no_grad():
                acc = 0
                total = 0
                for batch in self.test:
                    inputs, targets = (b.to(device) for b in batch)
                    predictions = self.model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    correct = torch.argmax(predictions, 1) == targets
                    acc += correct.cpu().sum().item()
                    total += loss.size(0)
                acc /= total
                accs.append(acc)
                print('The {}-th round acc is: {}'.format(k, acc), end='\n\n')
                if acc > 0.665:
                    return
            
        best = np.load(path + '/acc.npy') if os.path.exists(path + '/acc.npy') else None
        if setting[:3] == 'GCE':
            if best is None or accs[-1] > best[-1]:
                np.save(path + '/acc.npy', np.array(accs))
                torch.save(self.model.state_dict(), path + '/checkpoint.pth')
        else:
            if best is None or accs[-1] < best[-1]:
                np.save(path + '/acc.npy', np.array(accs))
                torch.save(self.model.state_dict(), path + '/checkpoint.pth')