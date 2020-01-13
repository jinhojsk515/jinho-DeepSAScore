import pickle
import time
import numpy as np
import torch.nn as nn
import torch
import random
import os
import argparse
import math
import model
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset import MolDataset, my_collate
import utils
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_feature", help="number of feature", type=int, default = 128)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=128)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 1)
parser.add_argument("--n_layer", help="layer of cell", type=int, default=1)
parser.add_argument("--n_sampling", help="count of sampling", type=int, default=1000)
parser.add_argument("--filenames", help="filenames", nargs='+', type=str)
parser.add_argument("--save_files", help="filenames", type=str)
parser.add_argument("--c_to_i", help="pickle file of c_to_i", type=str, default='data/c_to_i.pkl')
parser.add_argument("--i_to_c", help="pickle file of i_to_c", type=str, default='data/i_to_c.pkl')
args = parser.parse_args()
print(args)

if args.ngpu>0:
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c_to_i = pickle.load(open(args.c_to_i, 'rb'))
i_to_c = pickle.load(open(args.i_to_c, 'rb'))
n_char = len(c_to_i)

dataloaders = []

for fn in args.filenames:
    with open(fn) as f:
        lines = f.readlines()
        lines = [s.strip().split()[1] for s in lines]
        test_dataset = MolDataset(lines, c_to_i)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
    dataloaders.append(test_dataloader)
        

model = model.RNN(args.n_feature, args.n_feature, n_char, args.n_layer, i_to_c)

model = utils.initialize_model(model, device, args.save_files)

print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))

model.eval()
for fn,dataloader in zip(args.filenames, dataloaders):
    log_likelihoods = []
    for i_batch, sample in enumerate(dataloader) :
        x, l = sample['X'].to(device).long(), sample['L'].long().data.cpu().numpy()
        output, p_char = model(x) 
        p_char = torch.log(p_char+1e-10)
        p_char = p_char.data.cpu().numpy()

        x = x.data.cpu().numpy()
        for i in range(len(l)) :
            log_likelihood = 0
            for j in range(l[i]+1):
                log_likelihood += p_char[i,j,x[i,j]]
            log_likelihoods.append(log_likelihood)                
        break
    sns.distplot(log_likelihoods, label=fn)
#plt.ylim([0,0.02])
plt.legend()
plt.show()    
