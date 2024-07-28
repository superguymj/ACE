import torch
import torch.nn as nn
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in
        self.hidden_size = configs.seq_len * 2
        self.num_layers = 1
        self.output_size = configs.dec_in
        self.num_directions = 1
        self.LSTM = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.Linear = nn.Linear(self.hidden_size, self.output_size)
 
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]
        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.LSTM(x_enc, (h_0.detach(), c_0.detach())) 
        output = output[:, -self.pred_len:, :]
        output = self.Linear(output)
        return output
