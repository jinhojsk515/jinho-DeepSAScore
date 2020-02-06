import torch
import pickle
import time
import numpy as np
import torch.nn as nn
import torch
import random
import os
import argparse
import math
import models
from rdkit import Chem
#from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from torch.utils.data import DataLoader
from dataset import MolDataset, my_collate
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--n_head", help="number of head", type=int, default = 8)
parser.add_argument("--n_ff", help="number of feedforward feature", type=int, default = 2048)
parser.add_argument("--n_feature", help="number of feature", type=int, default = 128)
parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
parser.add_argument("--n_layer", help="layer of cell", type=int, default=4)
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--lr_decay", help="learning decay", type=float, default = 0.99)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=128)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 1)
parser.add_argument("--save_dir", help="save directory", type=str, default = 'save2')
parser.add_argument("--n_sampling", help="count of sampling", type=int, default=100)
parser.add_argument("--train_filenames", help="train filenames", type=str, default='data/pubchem/train.txt')
parser.add_argument("--test_filenames", help="test filenames", type=str, default='data/pubchem/test.txt')
parser.add_argument("--c_to_i", help="pickle file of c_to_i", type=str, default='data/pubchem/c_to_i.pkl')
parser.add_argument("--i_to_c", help="pickle file of i_to_c", type=str, default='data/pubchem/i_to_c.pkl')
parser.add_argument("--model", help="rnn or transformer", type=str, default='Trans')
parser.add_argument("--stereo", help="whether consider stereochemistry", type=bool, default = True)
parser.add_argument("--enumerate_smiles", help="enumerate smiles", type=bool, default = True)

args = parser.parse_args()
print(args)

if args.ngpu>0:
    cmd = utils.set_cuda_visible_device(args.ngpu)
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
else:
    os.environ['CUDA_VISIBLE_DEVICES']=''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_dir = args.save_dir

if not os.path.isdir(save_dir) :
    os.system('mkdir ' + save_dir)

c_to_i = pickle.load(open(args.c_to_i, 'rb'))
i_to_c = pickle.load(open(args.i_to_c, 'rb'))
n_char = len(c_to_i)

print ('c_to_i:', c_to_i)
with open(args.train_filenames) as f:
    lines = f.readlines()
    train_lines = [s.strip().split()[1] for s in lines]
train_dataset = MolDataset(train_lines, dict(c_to_i), args.enumerate_smiles, args.stereo)

with open(args.test_filenames) as f:
    lines = f.readlines()
    test_lines = [s.strip().split()[1] for s in lines]
test_dataset = MolDataset(test_lines, dict(c_to_i), args.enumerate_smiles, args.stereo)

train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=my_collate)

if args.model=='Trans':
    model = models.TransformerModel(args, n_char, i_to_c)
else:
    model = models.RNN(args, n_char, i_to_c)

model = utils.initialize_model(model, device)
print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("number of train_set :", len(train_lines))
print("number of test_set :", len(test_lines))
print("")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loss_fn = nn.CrossEntropyLoss(reduction='none')
min_loss = 10000

for epoch in range(args.epoch) :
    st = time.time()
    train_loss = []
    test_loss = []
    
    #train
    model.train()
    for i_batch, sample in enumerate(train_dataloader) :
        x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
        output = model(x)
        mask = utils.len_mask(l+1, output.size(1)-1)
        loss=torch.sum(loss_fn(output[:,:-1].reshape(-1, n_char), x.reshape(-1))*mask)/mask.sum()
        loss.backward()    
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        train_loss.append(loss.data.cpu().numpy())
        model.zero_grad()
    
    #test
    model.eval()
    with torch.no_grad():
        for i_batch, sample in enumerate(test_dataloader) :
            x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
            output = model(x) 
            mask = utils.len_mask(l+1, output.size(1)-1)
            loss=torch.sum(loss_fn(output[:,:-1].reshape(-1, n_char), x.reshape(-1))*mask)/mask.sum()
            test_loss.append(loss.data.cpu().numpy())

    #sampling
    n_generated=0
    n_in_train=0
    n_in_test=0

    with open(save_dir + '/generate', 'w') as f:
        for _ in range(args.n_sampling) :
            if args.ngpu>1 :
                result = model.module.sampling(120)
            else :
                result = model.sampling(120)
            result = ''.join([result])
            if Chem.MolFromSmiles(result) is not None :
                n_generated+=1
                if result in train_lines :
                    n_in_train+=1
                if result in test_lines:
                    n_in_test+=1
                f.write(result+'\n')
    end = time.time()

    train_loss = np.mean(np.array(train_loss))
    test_loss = np.mean(np.array(test_loss))
    print(f"{epoch:<6d}|{train_loss:9.6f} |{test_loss:9.6f} |{n_in_train:5d} |{n_in_test:5d} |{n_generated:5d} |{end-st:7.2f}\n") 
    
    if test_loss < min_loss :
        name = f'{save_dir}/save_{epoch}.pt'
        if args.ngpu>1:
            torch.save(model.module.state_dict(), name)
        else:
            torch.save(model.state_dict(), name)
        min_loss = test_loss
        
    for param_group in optimizer.param_groups :
        param_group['lr'] = args.lr * (args.lr_decay ** epoch)
