import os
import numpy as np
from bayes_opt import BayesianOptimization

model = 'GRU'

class MyBO():
    
    inf = 1e9
    
    def __init__(self, data):
        self.data = data
        self.BO = BayesianOptimization(
            self.TransferLearning,
            {
                "dropout": (0.01, 0.1),
                "learning_rate": (0.00001, 0.001),
            },
        )
        self.BO.maximize(init_points=5, n_iter=25)
        self.target = -self.BO.max['target']
        self.params = self.BO.max['params']
        
        
    def TransferLearning(self, dropout, learning_rate):
        ins = "python -u run.py --model {} --data {} --dropout {} --learning_rate {} --features MS --seq_len 32 --label_len 16 --pred_len 32 --freq s".format(model, self.data, dropout, learning_rate)
        os.system(ins)
        return -np.load(r'./results/{}/metrics.npy'.format(self.data))[0]
    
    def save(self, file, mode):
        with open(file, mode) as f:
            d = {
                'target': self.target,
                'params': self.params
            }
            f.write(str(d) + "\n")
            
class UserModel():
    
    def __init__(self, data, target, dropout, learning_rate):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.learning_rate = learning_rate
        
    def train(self, itr):
        ins = "python -u run.py --model {} --data {} --dropout {} --learning_rate {} --features MS --seq_len 32 --label_len 16 --pred_len 32 --freq s --itr {}".format(model, self.data, self.dropout, self.learning_rate, itr)
        os.system(ins)
        
    def test(self):
        ins = "python -u run.py --model {} --data {} --is_training 0 --features MS --seq_len 32 --label_len 16 --pred_len 32 --freq s".format(model, self.data)
        os.system(ins)
        

if __name__ == "__main__":
    
    n = 95
    
    os.system("python -u run.py --model {} --data Server --features MS --seq_len 32 --label_len 16 --pred_len 32 --freq s".format(model))
    
    with open(r"./loss/{}/hyperparameters.txt".format(model), "w") as f:
        f.write('')
    
    for i in range(n):
        data = "User{}".format(i)
        user = MyBO(data)
        user.save(r"./loss/{}/hyperparameters.txt".format(model), "a")
        print(data, user.BO.max)
    
    users = []
    with open(r"./hyperparameters.txt", "r") as f:
        lines = f.readlines()
        for (i, line) in enumerate(lines):
            d = dict(eval(line))
            user = UserModel('User{}'.format(i), d['target'], d['params']['dropout'], d['params']['learning_rate'])
            users.append(user)
    # users.sort(key=lambda user: user.target)
    # err = []
    # for i in range(n):
    #     print(users[i].data, users[i].target, sep=',')
    #     err.append(users[i].target)
    # print(np.mean(np.array(err))) 
            
    itr = 4
    for user in users:
        user.train(itr)
        user.test()
    
            
    metrics = [np.load(r"./results/Server/metrics.npy")]
    basic = [np.load(r"./results/Server/metrics.npy")]
            
    for i in range(n):
        user = users[i]
        copyPath = r"./results/" + '{}/'.format(user.data)
        metrics.append(np.load(copyPath + 'metrics.npy'))
        basic.append(np.load(copyPath + 'basic.npy'))
 
    np.save(r"./loss/{}/TL.npy".format(model), np.array(metrics))
    np.save(r"./loss/{}/nonTL.npy".format(model), np.array(basic))
        
    # path = r"C:/Projects/Federate learning/FL/UserInfo/"
    # models = []
    # for user in users:
    #     copyPath = r"./results/" + '{}/'.format(user.data)
        
    #     metrics = np.load(copyPath + 'metrics.npy')
    #     basic = np.load(copyPath + 'basic.npy')
    #     preds = np.load(copyPath + 'pred.npy')
    #     trues = np.load(copyPath + 'true.npy')
    #     Mn, Mx = np.load(copyPath + 'parameter.npy')
        
    #     models.append((metrics, basic, preds, trues, np.array([user.dropout, user.learning_rate, Mn, Mx])))

    # models.sort(key=lambda model: model[0][0])
       
    # for (i, model) in enumerate(models): 
    #     pastePath = path + 'User{}/'.format(i)
        
    #     metrics = model[0]
    #     basic = model[1]
    #     preds = model[2]
    #     trues = model[3]
    #     parameter = model[4]
        
    #     np.save(pastePath + 'metrics.npy', metrics)
    #     np.save(pastePath + 'basic.npy', basic)
    #     np.save(pastePath + 'pred.npy', preds)
    #     np.save(pastePath + 'true.npy', trues)
    #     np.save(pastePath + 'parameter.npy', parameter)
        