import os
import numpy as np
from bayes_opt import BayesianOptimization

if __name__ == "__main__":
    
    n = 95
    
    # for i in range(n):
    #     os.system("python -u run.py --model FEDformer --data User{} --features MS --seq_len 32 --label_len 16 --pred_len 32 --freq s".format(i))
    
    metrics = []
    
    for i in range(n):
        metrics.append(np.load('./results/User{}/metrics.npy'.format(i)))
        
    np.save('./loss/traditional/metrics.npy', np.array(metrics))