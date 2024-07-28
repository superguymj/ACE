import numpy as np
from sklearn.linear_model import LinearRegression
from tools import Metrics

class PGSP:
    fIdle = [(1 - 0.2) * 1, (1 - 0.5) * 1, (1 - 0.75) * 1]
    fOccp = [(1 - 0.3) * 1, (1 - 0.45) * 1, (1 - 0.6) * 1]
    reward = [0.25, 0.5, 0.75, 1]
    
    def __init__(self, bfis, sats):
        n = len(sats)
        m = 36
        w = [[1/3, 1/3] for i in range(n)]
        fIdle = PGSP.fIdle; fOccp = PGSP.fOccp; reward = PGSP.reward
        p = [[i, j, k] for i in fIdle for j in fOccp for k in reward]
        eta1, eta2 = 0, 1

        iter = 100
        for t in range(iter):    
            x = [w[i][0] * l[0] + w[i][1] * l[1] +
                (1 - w[i][0] - w[i][1]) * l[2] for i in range(n) for l in p]
            y = [sats[i].sat[j][k][l] for i in range(n) for j in range(
                3) for k in range(3) for l in range(4)]
            x, y = np.array(x).reshape((-1, 1)), np.array(y)
            model = LinearRegression().fit(x, y)
            eta1, eta2 = model.intercept_, model.coef_[0]
            for i in range(n):
                x = [[fIdle[j] - reward[l], fOccp[k] - reward[l]]
                    for j in range(3) for k in range(3) for l in range(4)]
                y = [(sats[i].sat[j][k][l] - eta1) / eta2 - reward[l]
                    for j in range(3) for k in range(3) for l in range(4)]
                x, y = np.array(x), np.array(y)
                model = LinearRegression(fit_intercept=False).fit(x, y)
                w[i] = list(model.coef_)
        
        # print(eta1, eta2)
        
        A = []
        x = []
        for i in range(n):
            x.append(bfis[i].vector)
        x = np.array(x)
        
        for col in range(2):
            y = []
            for i in range(n):
                y.append(w[i][col])
            y = np.array(y)
            model = LinearRegression(fit_intercept=False).fit(x, y)
            A.append(model.coef_)
            
        self.A = A
        self.eta1 = eta1; self.eta2 = eta2
            
    def pridict(self, bfis, sats):
        n = len(sats)
        fIdle = PGSP.fIdle; fOccp = PGSP.fOccp; reward = PGSP.reward
        err = Metrics()
        
        preds = []
        trues = []
        
        mean = [0, 0, 0, 0, 0]
        for i in range(5):
            for bfi in bfis:
                mean[i] += bfi.vector[i] / n
        y = [[] for i in range(10)]
        DoS = [[] for i in range(10)]
        
        for i in range(n):
            w0, w1 = bfis[i].weight(self)
            # print(w0, w1, 1 - w0 - w1)
            
            true = np.array(sats[i].sat)
            pred = np.array([[[0 for k in range(4)] for j in range(3)] for i in range(3)])
            for j in range(3):
                for k in range(3):
                    for l in range(4):
                        pred[j][k][l] = self.eta1 + self.eta2 * (w0 * fIdle[j] + w1 * fOccp[k] + (1 - w0 - w1) * reward[l])

            true = sats[i].scaler.inverse_transform(true)
            pred = sats[i].scaler.inverse_transform(pred)
            
            err.fit(pred, true)    
            
            
            pred = pred.flatten()
            true = true.flatten() 
            
            for x in pred:
                preds.append(x)
            for x in true:
                trues.append(x)
                
            error = pred - true
            for j in range(5):
                if bfis[i].vector[j] <= mean[j]:
                    for x in error:
                        y[j * 2].append(x)
                else:
                    for x in error:
                        y[j * 2 + 1].append(x)
                        
            for (a, b) in zip(pred, true):
                DoS[min(10, max(1, int(b + 0.5))) - 1].append(min(10, max(1, a)))
                    
        mae, mse, rmse, mape, mspe = err.metrics()
        print("mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}".format(mae, mse, rmse, mape, mspe))
        err.show()
        
        return preds, trues, y, DoS