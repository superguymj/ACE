import numpy as np


class StandardScaler():
    eps = 1e-9

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data) + StandardScaler.eps

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class Metrics():
    def __init__(self):
        self.mae = 0
        self.mse = 0
        self.rmse = 0
        self.mape = 0
        self.mspe = 0
        self.count = [0 for i in range(11)]
        self.matrix = [[0 for j in range(11)] for i in range(11)]
    
    def fit(self, pred, true):
        pred = pred.flatten()
        true = true.flatten()
        self.mae += self.MAE(pred, true)
        self.mse += self.MSE(pred, true)
        self.rmse += self.RMSE(pred, true)
        self.mape += self.MAPE(pred, true)
        self.mspe += self.MSPE(pred, true)
        self.count[0] += 1
        for (x, y) in zip(pred, true):
            x = min(10, max(1, int(round(x, 0))))
            y = min(10, max(1, int(round(y, 0))))
            self.count[y] += 1
            self.matrix[y][x] += 1
        
        
    def MAE(self, pred, true):
        return np.mean(np.abs(pred - true))

    def MSE(self, pred, true):
        return np.mean((pred - true) ** 2)


    def RMSE(self, pred, true):
        return np.sqrt(self.MSE(pred, true))


    def MAPE(self, pred, true):
        return np.mean(np.abs((pred - true) / true))


    def MSPE(self, pred, true):
        return np.mean(np.square((pred - true) / true))
    
    def metrics(self):
        cnt = self.count[0]
        return self.mae / cnt, self.mse / cnt, self.rmse / cnt, self.mape / cnt, self.mspe / cnt
    
    def show(self):
        for i in range(1, 11):
            for j in range(1, 11):
                tmp = self.matrix[i][j] / self.count[i]
                print('{:2d}'.format(int(tmp * 100)), end=' ')
            print('')