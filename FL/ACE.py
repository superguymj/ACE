import numpy as np
import random
import time
import FL

from utility.step_lr import StepLR

eps = 1e-30
inf = 1e30


class Client(FL.Client):

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
        super(Client, self).__init__(id, freq, B, p, h, Xi, Battery, bfi, sat, model, pred, true, trainData, testData)


class Server(FL.Server):
    Beta = 0.99
    T = 10
    N = 100
    e = 100
    base_lr = 0.1

    def __init__(self, clients, Deadline, Limit_avg, testDatas):
        super(Server, self).__init__(clients, Deadline, Limit_avg, testDatas)

        self.temp = [clients[i] for i in range(self.n)]
        self.temp.sort(key=lambda client: client.sens, reverse=True)
        self.tabLr = [self.lr(i) for i in range(Server.K)]
    
    def split(self, Deadline):
        t = [client.shortest for client in self.clients]
        t.sort()
        limit = t[self.n // 7]
        
        Deadline -= limit * Server.K
        w = [Server.Beta ** i for i in range(Server.K)]
        
        W = sum(w)
        
        Deadlines = [Deadline * w[i] / W + limit for i in range(Server.K)]
        return Deadlines
    
    def lr(self, k):
        if k < Server.K * 3/10:
            return Server.base_lr
        elif k < Server.K * 6/10:
            return Server.base_lr * 0.2
        elif k < Server.K * 8/10:
            return Server.base_lr * 0.2 ** 2
        else:
            return Server.base_lr * 0.2 ** 3

    def Reward(self, selection, Deadlines):
        
        tabFreq = [[self.clients[j].Train(Deadlines[i]) 
                         for j in range(self.n)] for i in range(Server.K)]
        tabReward = [[self.clients[j].Reward(self.clients[j].pred[i], tabFreq[i][j][0], 0)
                           for j in range(self.n)] for i in range(Server.K)]
        tabEnergy = [[self.clients[j].Energy(self.clients[j].pred[i], tabFreq[i][j][1], Deadlines[i])
                           for j in range(self.n)] for i in range(Server.K)]
        tabSat = [[self.clients[j].Satisfaction(self.clients[j].pred[i], tabFreq[i][j][0], tabReward[i][j])
                        for j in range(self.n)] for i in range(Server.K)]
        
        freq = [[0 for j in range(self.n)] for i in range(Server.K)]
        reward = [[0 for j in range(self.n)] for i in range(Server.K)]
        acc = 0
        cnt = [self.clients[i].D for i in range(self.n)]
        
        for i in range(Server.K):
            count = sum(selection[i])
            if count == 0:
                continue
            sat = [0 for j in range(self.n)]
            for j in range(self.n):
                if selection[i][j] == True:
                    if tabFreq[i][j][0] is None:
                        return None, None, None, -inf
                    freq[i][j] = tabFreq[i][j][1]
                    reward[i][j] = tabReward[i][j]
                    sat[j] = tabSat[i][j]
                    acc += cnt[j] * self.tabLr[i]
                    cnt[j] *= Server.alpha
            totSat = sum(sat)
            for client in self.temp:
                if totSat >= self.Limit_avg * count:
                    break
                if selection[i][client.id] == False:
                    continue
                id = client.id
                need = self.Limit_avg * count - totSat
                totSat -= sat[id]
                if sat[id] + need <= 10:
                    sat[id] += need
                else:
                    sat[id] = 10
                totSat += sat[id]
                prec = tabFreq[i][id][0]
                reward[i][id] = client.Reward(client.pred[i], prec, sat[id])
        
        totReward = sum(sum(reward, []))
        E = [0 for i in range(self.n)]
        for j in range(self.n):
            for i in range(Server.K):
                E[j] += tabEnergy[i][j][int(selection[i][j])]
            E[j] /= self.clients[j].Battery
        E = np.array(E)
        if totReward > self.Budget or E.max() > 1.0:
            over = min(0, self.Budget - totReward) - np.sum(E[E > 1.0])
            return None, None, None, over
        return selection, freq, reward, acc

    def clientSelection(self):
        random.seed(19260817)
        p = [[0.90 - 0.30 * i / Server.K for j in range(self.n)] for i in range(Server.K)]
        # l = [self.Deadline / Server.K for i in range(Server.K - 1)]
        selection, acc = None, 0
        for _ in range(Server.T):
            print('start iteration {}:'.format(_ + 1))
            start = time.time()
            examples = []
            id = list(range(Server.N))
            for __ in range(Server.N):
                s = [[random.choices([False, True], cum_weights=[1 - p[i][j], 1])[0]
                      for j in range(self.n)] for i in range(Server.K)]
                # t = [random.expovariate(1.0 / l[i]) for i in range(Server.K - 1)]
                # while sum(t) > self.Deadline:
                #     t = [random.expovariate(1.0 / l[i]) for i in range(Server.K - 1)]
                # t.append(self.Deadline - sum(t))
                
                t = self.split(self.Deadline)
                
                for i in range(Server.K):
                    for j in range(self.n):
                        if self.clients[j].shortest > t[i]:
                            s[i][j] = False
                examples.append((s, t, self.Reward(s, t)[3]))
            id.sort(key=lambda i: examples[i][1], reverse=True)

            if examples[id[0]][2] > acc:
                selection, Deadlines, acc = examples[id[0]]
                print(sum(selection, np.array([0] * self.n)))
                print([sum(i) for i in selection])
            print(acc)

            temps = np.array([[0 for j in range(self.n)]
                            for i in range(Server.K)])
            tempt = np.array([0.0 for i in range(Server.K)])
            for i in range(Server.e):
                temps += np.array(examples[id[i]][0])
                tempt += np.array(examples[id[i]][1])
            p = temps / Server.e
            l = (tempt / Server.e)[:-1]

            print('time: {}s'.format(time.time() - start))

        return self.Reward(selection, Deadlines)
