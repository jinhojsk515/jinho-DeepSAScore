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

parser = argparse.ArgumentParser()
parser.add_argument("--n_head", help="number of head", type=int, default = 8)
parser.add_argument("--n_ff", help="number of feedforward feature", type=int, default = 2048)
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0005)
parser.add_argument("--lr_decay", help="learning decay", type=float, default = 0.99)
parser.add_argument("--n_feature", help="number of feature", type=int, default = 128)
parser.add_argument("--epoch", help="epoch", type=int, default = 10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=128)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 1)
parser.add_argument("--save_dir", help="save directory", type=str, default = 'save')
parser.add_argument("--n_layer", help="layer of cell", type=int, default=1)
parser.add_argument("--n_sampling", help="count of sampling", type=int, default=1000)
parser.add_argument("--train_filenames", help="train filenames", type=str, default='data/train.txt')
parser.add_argument("--test_filenames", help="test filenames", type=str, default='data/test.txt')
parser.add_argument("--c_to_i", help="pickle file of c_to_i", type=str, default='data/c_to_i.pkl')
parser.add_argument("--i_to_c", help="pickle file of i_to_c", type=str, default='data/i_to_c.pkl')
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
    train_lines = [s.strip().split()[1] for s in lines][:1]
train_dataset = MolDataset(train_lines, dict(c_to_i))#train_lines

with open(args.test_filenames) as f:
    lines = f.readlines()
    test_lines = [s.strip().split()[1] for s in lines][:1]
test_dataset = MolDataset(test_lines, dict(c_to_i))#test_lines

train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)

#model = model.RNN(args.n_feature, args.n_feature, n_char, args.n_layer, i_to_c)
model = model.TransformerModel(n_char, args.n_feature, args.n_head, args.n_feature, args.n_layer, args.batch_size, args.n_ff, i_to_c)
model = utils.initialize_model(model, device)
print("number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loss_fn = nn.CrossEntropyLoss(reduction='mean')
min_loss = 10000

for epoch in range(args.epoch) :
    st = time.time()
    train_loss = []
    test_loss = []

    model.train()
    for i_batch, sample in enumerate(train_dataloader) :
        x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
        output, _ = model(x)
        loss=0
        for i in range(len(l)) :
            loss += loss_fn(output[i, :l[i]+1, :], x[i, :l[i]+1])
        loss = loss/len(l)
        loss.backward()    
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        model.zero_grad()
        train_loss.append(loss.data.cpu().numpy())
        #if i_batch>1: break
    model.eval()
    for i_batch, sample in enumerate(test_dataloader) :
        x, l = sample['X'].to(device).long(), sample['L'].to(device).long()
        output, _ = model(x) 
        loss=0
        for i in range(len(l)) :
            loss += loss_fn(output[i, :l[i]+1, :], x[i, :l[i]+1])
        loss = loss/len(l)
        test_loss.append(loss.data.cpu().numpy())
        #if i_batch>1: break

    #sampling
    n_generated=0
    n_in_data=0
    for a in range(args.n_sampling) :
        result, p = model.sampling(120)
        result = ''.join([result])
        try :
            if Chem.MolFromSmiles(result) is not None :
                n_generated+=1
        except :
            pass
        if result in train_lines :
            n_in_data+=1
    end = time.time()

    train_loss = np.mean(np.array(train_loss))
    test_loss = np.mean(np.array(test_loss))
    print(f"{epoch}\t{train_loss:.3f}\t{test_loss:.3f}\t{n_in_data}\t{n_generated}\t{end-st}")
    
    if test_loss<min_loss :
        name = f'{save_dir}/save_{epoch}.pt'
        torch.save(model.state_dict(), name)
        min_loss = test_loss
    for param_group in optimizer.param_groups :
        param_group['lr'] = args.lr * (args.lr_decay ** epoch)
