import os
import torch
import argparse
import numpy as np
import pandas as pd
from BFI_2.BigFiveInventory import BFI_2
from BFI_2.Satisfaction import Satis
from BFI_2.Model import PGSP
import FL, ACE
from data import Dataset_CIFAR10, Dataset_CIFAR100, Dataset_MNIST, Dataset_FMNIST, Dataset_FLOWERS102, Dataset_EUROSAT
from model.wide_res_net import WideResNet
from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Bs, ps, hs, Xis, Batterys = np.load('./UserInfo/parameters.npy')

selectionParser = {
    'ACE': (ACE.Server, ACE.Client),
}
modelParser = {
    'wideResnet': WideResNet,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
}

datasetParser = {
    'cifar10': Dataset_CIFAR10,
    'cifar100': Dataset_CIFAR100,
    'mnist': Dataset_MNIST,
    'fmnist': Dataset_FMNIST,
    'flowers102': Dataset_FLOWERS102,
    'eurosat': Dataset_EUROSAT,
}

def createClients(n, Client, args):
    clients = []; freqs = []; preds = []; trues = []
    for i in range(n):
        path = r'./UserInfo/User{}/'.format(i)
        _, __, Mn, Mx = np.load(path + 'parameter.npy')
        freqs.append((Mn, Mx))
        preds.append(np.load(path + 'pred.npy'))
        trues.append(np.load(path + 'true.npy'))
        
    
    data = pd.read_csv("./BFI_2/Questionnaire.csv", header=None)
    data = data.values
    # print(len(data))
    bfis = []
    sats = []
    for i in range(n):
        bfis.append(BFI_2(data[i][36:]))
        sats.append(Satis(data[i][:36]))
    model = PGSP(bfis, sats)
    model.pridict(bfis, sats)    
    
        
    Dataset = datasetParser[args.dataset]
        
    trainDatas = Dataset(); trainDatas.load(train=True); trainDatas = trainDatas.split(n, iid=(args.split_type == 'iid'))
    testDatas = Dataset(); testDatas.load(train=False); testDatas = testDatas.split(n, iid=(args.split_type == 'iid'))
    
    if args.dataset == 'eurosat':
        global Batterys
        Batterys = Batterys / 3.0
    
    for (id, freq, B, p, h, Xi, Battery, bfi, sat, pred, true, trainData, testData) in zip(list(range(n)), freqs, Bs, ps, hs, Xis, Batterys, bfis, sats, preds, trues, trainDatas, testDatas):
        clients.append(Client(id, freq, B, p, h, Xi, Battery, bfi, sat, model, pred, true, trainData, testData))
        
    return clients

def createServer(n, Server, Client, args):
    clients = createClients(n, Client, args)
    Limit_avg = 8.5
    
    Deadline = FL.Server.K * 3 * 60
    if args.dataset == 'eurosat':
        Deadline = FL.Server.K * 1 * 60
        
    Dataset = datasetParser[args.dataset]
    testDatas = Dataset(); testDatas.load(train=False)
    return Server(clients, Deadline, Limit_avg, testDatas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default='ACE', type=str, help="Clients selection model, options: [ACE, KDP, PDG, Classic, SL]")
    parser.add_argument("--model", default='wideResnet', type=str, help="Federate learning model, options: [wideResnet, vgg11, vgg13, vgg16, vgg19, ]")
    parser.add_argument("--clients", default=50, type=int, help="Number of clients")
    parser.add_argument("--rounds", default=192, type=int, help="Number of training rounds")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs for each client in one training round")
    parser.add_argument("--dataset", default='cifar10', type=str, help="Dataset type, options: [cifar10, cifar100, mnist, ]")
    parser.add_argument("--split_type", default='noniid', type=str, help="Split type of Dataset, options: [noniid, iid]")
    args = parser.parse_args()

    setting = '{}_{}_{}clients_{}rounds_{}epochs_{}_{}'.format(args.selection, args.model, args.clients, args.rounds, args.epochs, args.dataset, args.split_type)
    path = './results/{}'.format(setting)
    if not os.path.exists(path):
        os.makedirs(path)

    n = args.clients
    FL.Server.K = args.rounds
    FL.Client.epochs = args.epochs
    Server, Client = selectionParser[args.selection]
    FL.Server.Model = modelParser[args.model]
    server = createServer(n, Server, Client, args)

    selection, freq, reward, acc = server.clientSelection()
    selection = np.array(selection)
    np.save(path + '/selection.npy', selection)

    selection = np.load(path + '/selection.npy', allow_pickle=True)
    print(sum(selection, np.array([0] * n)))
    print([sum(i) for i in selection])
    for i in range(3000):
        server.initModel()
        server.train(selection, setting)
    
if __name__ == "__main__":
    main()

        