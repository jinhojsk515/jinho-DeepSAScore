import pickle
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import os
import argparse
import math
import models
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset import MolDataset, my_collate
import utils
#import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Contrib.SA_Score.sascorer import calculateScore
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters, CalcNumUnspecifiedAtomStereoCenters
from scipy import stats
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_feature", help="number of feature", type=int, default = 1024)
parser.add_argument("--n_ff", help="number of heads", type=int, default=1024)
parser.add_argument("--n_layer", help="layer of cell", type=int, default=4)
parser.add_argument("--n_head", help="number of head", type=int, default=8)
parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 1)
parser.add_argument("--save_files", help="filenames", type=str)
parser.add_argument("--c_to_i", help="pickle file of c_to_i", type=str, default='data/stock/c_to_i.pkl')
parser.add_argument("--i_to_c", help="pickle file of i_to_c", type=str, default='data/stock/i_to_c.pkl')
parser.add_argument("--model", help="rnn or transformer", type=str, default='Trans')
parser.add_argument("--stereo", help="whether consider stereochemistry", type=bool, default = True)

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

with open('data/testing.txt') as f:
    lines = f.readlines()
    lines = [l.strip().split('\t') for l in lines]
    s_to_human_score = {l[1]:l[2] for l in lines}

if args.model=='Trans':
    model = models.TransformerModel(args, n_char, i_to_c)
else:
    model = models.RNN(args, n_char, i_to_c)

model = utils.initialize_model(model, device, args.save_files)

print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))

model.eval()

log_likelihoods = []
synthesis = []
sascores =[]
ok_sascore=[]
no_sascore=[]
ok_ourscore=[]
no_ourscore=[]

with torch.no_grad() :
    for s in s_to_human_score.keys():
    #for s in ['COc1ccc2cc1-c1c(O)cc(O)c3c(=O)cc(oc13)-c1ccc(O)c(c1)[C@H](C)c1c(O)cc3c(c1O)C(=O)C[C@@H]2O3']:
        m = Chem.MolFromSmiles(s)
        num_sc=CalcNumAtomStereoCenters(m)-CalcNumUnspecifiedAtomStereoCenters(m)
        synthesis.append(int(s_to_human_score[s]))
        sascores.append(calculateScore(m))

        if args.stereo :
            isomers=list(EnumerateStereoisomers(m))
        else : 
            isomers = [m]
        
        likelihood=[]
        for s in isomers:
            s=Chem.MolToSmiles(s, isomericSmiles=args.stereo)    
            s = s+'Q'
            x = torch.tensor([c_to_i[i] for i in list(s)]).unsqueeze(0).to(device)
            output=model(x)
            likelihood.append(utils.output_to_likelihood(output, x))
        a=max(likelihood)
         
        #log_likelihoods.append((np.log(-a))*(1+np.log(num_sc+1)))
        log_likelihoods.append((np.log(-a)))
        print(s, log_likelihoods[-1], synthesis[-1])

m=-np.array(sascores)
n=-np.array(log_likelihoods)

print()
print("sascore\t", roc_auc_score(synthesis,m))
print("ours\t", roc_auc_score(synthesis,n))
print()

synthesis, sascores, log_likelihoods = zip(*sorted(zip(synthesis, sascores, log_likelihoods)))
x = np.array(synthesis)
y1 = np.array(sascores)
y2 = np.array(log_likelihoods)

plt.scatter(x,y1,label='sascore',c='red',s=5)
plt.scatter(x,y2,label='model',c='green',s=5)
plt.ylabel('score')
plt.xlim([-1,2])
plt.legend()
plt.show()    


