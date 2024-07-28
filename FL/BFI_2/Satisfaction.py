import numpy as np
from tools import StandardScaler

class Satis:
    def __init__(self, Q):
        self.scaler = StandardScaler()
        self.scaler.fit(Q)
        Q = self.scaler.transform(Q)
        
        self.sat = np.array([[[0.0 for k in range(4)]
                     for j in range(3)] for i in range(3)])
    
        
        for i in range(3, -1, -1):
            for j in range(3):
                for k in range(3):
                    self.sat[j][k][i] = Q[0]
                    Q = Q[1:]
