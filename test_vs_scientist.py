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
#import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Contrib.SA_Score.sascorer import calculateScore
from scipy import stats
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from scipy.stats import gaussian_kde
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--enumerate_smiles", help="enumerate smiles", type=bool, default = True)
parser.add_argument("--stereo", help="whether consider stereochemistry", type=bool, default =True)
parser.add_argument("--n_feature", help="number of feature", type=int, default = 1024)
parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 1)
parser.add_argument("--n_layer", help="layer of cell", type=int, default=4)
parser.add_argument("--save_files", help="filenames", type=str)
parser.add_argument("--c_to_i", help="pickle file of c_to_i", type=str, default='data/stock/c_to_i.pkl')
parser.add_argument("--i_to_c", help="pickle file of i_to_c", type=str, default='data/stock/i_to_c.pkl')
parser.add_argument("--n_head", help="number of head", type=int, default=8)
parser.add_argument("--model", help="rnn or transformer", type=str, default='Trans')
parser.add_argument("--n_ff", help="number of heads", type=int, default=1024)

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

with open('data/vs_chemist.txt') as f:
    lines = f.readlines()
    lines = [l.strip().split() for l in lines]
    s_to_human_score = {l[1]:float(l[3]) for l in lines}

if args.model=='Trans':
    model = model.TransformerModel(args, n_char, i_to_c)
else:
    model = model.RNN(args, n_char, i_to_c)

model = utils.initialize_model(model, device, args.save_files)

print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
softmax=nn.Softmax(dim=-1)
model.eval()
log_likelihoods = []
humanscores = []
sascores =[]

with torch.no_grad() :
    for s in s_to_human_score.keys():
        humanscores.append(s_to_human_score[s])
        s = Chem.MolFromSmiles(s)
        sascores.append(calculateScore(s))
        if args.stereo :
            isomers=list(EnumerateStereoisomers(s))
        else : 
            isomers = [s]
        likelihood=[]
        for s in isomers:
            s=Chem.MolToSmiles(s, isomericSmiles=True)    
            s = s+'Q'
            x = torch.tensor([c_to_i[i] for i in list(s)]).unsqueeze(0).to(device)
            output=model(x)
            likelihood.append(utils.output_to_likelihood(output, x))
        a=max(likelihood)
        log_likelihoods.append(np.log(-a))
print()
humanscores, sascores, log_likelihoods = zip(*sorted(zip(humanscores, sascores, log_likelihoods)))
x = np.arange(0, len(sascores))
y1 = np.array(log_likelihoods)*2-4.5
y2 = np.array(sascores)
y3 = np.array(humanscores)

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y1)
print('\nours vs chemist')
print (f'rvalue: {r_value}\npvalue: {p_value}')
print (f'r2 value: {r_value*r_value}')

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y2)
print('\nsascores vs chemist')
print (f'rvalue: {r_value}\npvalue: {p_value}')
print (f'r2 value: {r_value*r_value}')

slope, intercept, r_value, p_value, std_err = stats.linregress(y1,y2)
print('\nsascores vs ours')
print (f'rvalue: {r_value}\npvalue: {p_value}')
print (f'r2 value: {r_value*r_value}')

print()
plt.plot(x, y1, label='ours', c='red')
plt.plot(x, y2, label='sascores', c='blue')
plt.plot(x, y3, label='chemist', c='green')
#plt.scatter(y1, y2)
plt.ylabel('Score')
#plt.ylim([-30,-10])
plt.legend()
plt.show()   

