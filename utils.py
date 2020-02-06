import numpy as np
import torch
from rdkit import Chem
import time
import glob
#from sklearn.metrics import euclidean_distances
from torch.autograd import Variable
#from Bio.PDB import *
import torch.nn as nn
import copy
import math

#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc

distance_criteria = 5.0
N_atom_features = 28

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=350):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def output_to_likelihood(output, x):
    result=0
    p_char=nn.LogSoftmax(dim=-1)(output)
    p_char = p_char.data.cpu().numpy()
    x = x.data.cpu().numpy()
    for i in range(output.size(1)-1):
        print(np.exp(p_char[0,i,x[0,i]]))
        result+=p_char[0,i,x[0, i]]
    return result

def p_cal(output, x):                                           
    result=[]                                                                 
    p_char=nn.LogSoftmax(dim=-1)(output)                                       
    #p_char=output                                       
    p_char = p_char.data.cpu().numpy()                                         
    x = x.data.cpu().numpy()                                                   
    for i in range(output.size(1)-1):
        result.append(np.exp(p_char[0,i,x[0, i]]))
        #result.append(p_char[0,i,x[0, i]])
    return result

def prob_to_smiles(output, x, i_to_c, is_sampling=True) :
    for i in range(2) :
        original = x[i]
        retval=output[i]
        org_result = ''
        result=''
        possibility = 0
        p_letter = nn.Softmax(-1)(retval)
        log_p_letter = -torch.log(p_letter)
        if is_sampling:
            retval = torch.distributions.categorical.Categorical(p_letter)
            retval = retval.sample()
        else :
            retval = retval.max(-1)[1]
            print(retval)

        for j in range(len(retval)) :
            codon = retval[j].item()
            possibility += log_p_letter[j][codon].item()
            codon = i_to_c[codon]
            if codon=='Q' :
                break
            result+=codon
        for j in range(len(original)) :
            codon = i_to_c[original[j].item()]
            if codon=='Q' :
                break
            org_result+=codon
        print('orginal : ' + org_result)
        print('sampled : ' + result)
        print('p_char  :', possibility)

def len_mask(l, max_length) :
    device = l.device
    mask = torch.arange(0, max_length).repeat(l.size(0)).to(device)
    l = l.unsqueeze(1).repeat(1, max_length).reshape(-1)
    mask = mask-l
    mask[mask>=0]=0
    mask[mask<0]=1
    return mask

def create_var(tensor, requires_grad=None): 
    if requires_grad is None: 
        #return Variable(tensor)
        return Variable(tensor).cuda()
    else: 
        return Variable(tensor,requires_grad=requires_grad).cuda()

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            if param.grad is None:
                continue
            shared_param._grad = param.grad.cpu()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_feature(m, atom_i):

    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
def cal_distance_matrix(m):
    c = m.GetConformers()[0]
    d = c.GetPositions()
    return euclidean_distances(d)

def process_protein(filename):
    parser = PDBParser()
    structure = parser.get_structure('site', filename)
    """
    center = []
    for atom in structure.get_atoms():
        center.append(atom.get_coord())
    center = np.mean(np.array(center),0)
    """
    #print (center)
    amino_acids = ['ALA','ARG','ASN','ASP','ASX','CYS','GLU','GLN','GLX',\
                   'GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',\
                   'THR','TRP','TYR','VAL']
    features = []
    positions = []
    
    for atom in structure.get_atoms():
        if atom.get_parent().get_resname() not in amino_acids : continue
        if atom.get_id()=='CA':
            p = atom.get_coord()
            positions.append(np.copy(p))
            features.append(amino_acids.index(atom.get_parent().get_resname()))
        elif atom.get_id()=='CB':
            p = atom.get_coord()
            positions.append(np.copy(p))
            features.append(amino_acids.index(atom.get_parent().get_resname())+len(amino_acids))
    positions = np.array(positions)
    features = np.eye(44)[features]
    #if len(features)>120: return None, None

    return positions, features

def preprocessor(pdbs, device, prefix):
    ligand_list = []
    p = pdbs[0]

    try:
        m1_ref = Chem.SDMolSupplier(prefix+p+'/ligand.sdf')[0]
        m1_sampled = Chem.SDMolSupplier(prefix+p+'/ligand_uff.sdf')[0]
        m2_site = Chem.MolFromPDBFile(prefix+p+'/site.pdb')
    except:
        return None
    if m1_ref is None: return None
    if m1_sampled is None: return None
    if m2_site is None: return None
    #if m1 is None:          m1 = Chem.MolFromMol2File(prefix+p+'/ligand.mol2')

    ligand_list.append((m1_ref, m1_sampled))
    c1_sampled = m1_sampled.GetConformers()[0]
    m1_sampled_positions = c1_sampled.GetPositions()
    c1_ref = m1_ref.GetConformers()[0]
    m1_ref_positions = c1_ref.GetPositions()

    p_positions, p_features = process_protein(prefix+p+'/site.pdb')
    if p_positions is None: return None
    n_l_atoms = m1_ref.GetNumAtoms()
    dm_ref = euclidean_distances(np.concatenate([m1_ref_positions, p_positions], 0))
    dm_sampled = euclidean_distances(np.concatenate([m1_sampled_positions, p_positions], 0))

    dm1_l_l = convert_to_onehot(dm_sampled[:n_l_atoms, :n_l_atoms], 0.0, 15.0, 0.5)
    dm1_l_p = convert_to_onehot(dm_sampled[:n_l_atoms, n_l_atoms:], 5, 17.0, 0.5)
    dm1_p_p = convert_to_onehot(dm_sampled[n_l_atoms:, n_l_atoms:], 0.0, 25.0, 0.5)
    dm2_l_l = convert_to_onehot(dm_ref[:n_l_atoms, :n_l_atoms], 0.0, 15.0, 0.5)
    dm2_l_p = convert_to_onehot(dm_ref[:n_l_atoms, n_l_atoms:], 5.0, 17.0, 0.5)

    l_features = []
    for i in range(m1_ref.GetNumAtoms()):
        l_features.append(atom_feature(m1_ref, i))
    l_features = np.array(l_features)*1
    dm1_l_l = create_var(torch.from_numpy(dm1_l_l)).to(device).float().unsqueeze(0)
    dm1_l_p = create_var(torch.from_numpy(dm1_l_p)).to(device).float().unsqueeze(0)
    dm1_p_p = create_var(torch.from_numpy(dm1_p_p)).to(device).float().unsqueeze(0)
    dm2_l_l = create_var(torch.from_numpy(dm2_l_l)).to(device).float().unsqueeze(0)
    dm2_l_p = create_var(torch.from_numpy(dm2_l_p)).to(device).float().unsqueeze(0)
    l_features = create_var(torch.from_numpy(l_features)).to(device).float().unsqueeze(0)
    p_features = create_var(torch.from_numpy(p_features)).to(device).float().unsqueeze(0)
    return l_features, p_features, dm1_l_l, dm1_l_p, dm1_p_p, dm2_l_l, dm2_l_p, ligand_list


def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(4):
        command = ['nvidia-smi','-i',str(i)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = str(p.communicate()[0])
        count = result.count('No running')
        if count>0:    
            empty.append(i)
    
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def make_ring_matrix(m):
    ssr = Chem.GetSymmSSSR(m)
    natoms = m.GetNumAtoms()
    retval = np.zeros((natoms, natoms))
    for indice in ssr:
        for i1 in indice:
            for i2 in indice:
                retval[i1,i2] = 1
    #print (retval)
    return retval                

def write_xyz(m, p, name):
    w = open(name, 'w')
    natoms = m.GetNumAtoms()
    w.write(str(natoms) + '\n\n')
    for i in range(natoms):
        atom = m.GetAtomWithIdx(i)
        w.write(atom.GetSymbol() + '\t' + str(p[i,0]) + '\t' + str(p[i,1]) + '\t' + str(p[i,2]) + '\n')
    w.close()

def probability_to_distance(prob):
    shape = list(np.shape(prob))
    n_classes = shape[-1]
    #indice = np.array([i+0.0 for i in range(n_classes)])
    indice = np.array([i+0.5 for i in range(n_classes)])
    shape = shape[:-1] + [1]
    indice = np.tile(indice, shape)
    retval = np.sum(indice*prob, -1)
    return retval


def optimize_geometry(position, target1, target2, protein_position, all_protein_position, around_indice1, around_indice2, adj, initial_dm):
    optimizer = torch.optim.Adam([position], lr=0.1)
    n_l_atoms = position.size(0)
    n_p_atoms = protein_position.size(0)
    n_all_p_atoms = all_protein_position.size(0)
    for i in range(500):
        optimizer.zero_grad()
        position1 = position.unsqueeze(1).repeat(1, n_l_atoms, 1)
        position2 = position.unsqueeze(0).repeat(n_l_atoms, 1, 1)
        position3 = position.unsqueeze(1).repeat(1, n_p_atoms, 1)
        position4 = protein_position.unsqueeze(0).repeat(n_l_atoms, 1, 1)
        position5 = position.unsqueeze(1).repeat(1, n_all_p_atoms, 1)
        position6 = all_protein_position.unsqueeze(0).repeat(n_l_atoms, 1, 1)

        dm1 = torch.sqrt(torch.sum(torch.pow(position1-position2,2), -1)+1e-6)
        dm2 = torch.sqrt(torch.sum(torch.pow(position3-position4,2), -1)+1e-6)
        dm3 = torch.sqrt(torch.sum(torch.pow(position5-position6,2), -1)+1e-6)
        
        loss1 = torch.sum(torch.pow((dm1-target1)*around_indice1,2))/around_indice1.sum()
        #loss2 = torch.mean(torch.pow((dm2-target2)*around_indice2,2))
        loss2 = torch.sum(torch.pow((dm2-target2)*around_indice2,2))/around_indice2.sum()
        loss3 = torch.mean((1.5-dm3).clamp(0.0))
        loss4 = torch.mean(torch.pow((dm1-initial_dm)*adj,2))
        
        loss = loss1*0.0+loss2*1.0+loss3*100+loss4*10.0
        #loss = loss1*10+loss2*1.0+loss3*100+loss4*10.0
        #loss = loss1*1.0+loss2*0.2+loss3*10.0

        loss.backward()
        optimizer.step()
        #if i%100==0 : 
        #    print (i, loss2.data.cpu().numpy())
    new_position = position.data.cpu().numpy()
    return new_position, loss1, loss2

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file)) 
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def convert_to_onehot(matrix, min_d, max_d, scaling):
    matrix[matrix>max_d] = max_d
    matrix[matrix<min_d] = min_d
    matrix -= min_d
    matrix /= scaling
    n_classes = int((max_d-min_d)/scaling+1)
    matrix = np.around(matrix).astype(int)
    retval = np.eye(n_classes)[matrix]
    
    return retval

def preprocessor2(data, device):
    l_features = np.array([np.copy(d[0]) for d in data])
    p_features = np.array([np.copy(d[1]) for d in data])
    dm1_l_l = np.array([convert_to_onehot(np.copy(d[2]), 0.0, 15.0, 0.5) for d in data])
    dm1_l_p = np.array([convert_to_onehot(np.copy(d[3]), 5.0, 17.0, 0.5) for d in data])
    dm1_p_p = np.array([convert_to_onehot(np.copy(d[4]), 0.0, 25.0, 0.5) for d in data])
    dm2_l_l = np.array([convert_to_onehot(np.copy(d[5]), 0.0, 15.0, 0.5) for d in data])
    dm2_l_p = np.array([convert_to_onehot(np.copy(d[6]), 5.0, 17.0, 0.5) for d in data])

    if len(dm1_l_l) ==0 : return None
    
    dm1_l_l = create_var(torch.from_numpy(dm1_l_l)).to(device).float()
    dm1_l_p = create_var(torch.from_numpy(dm1_l_p)).to(device).float()
    dm1_p_p = create_var(torch.from_numpy(dm1_p_p)).to(device).float()
    dm2_l_l = create_var(torch.from_numpy(dm2_l_l)).to(device).float()
    dm2_l_p = create_var(torch.from_numpy(dm2_l_p)).to(device).float()
    l_features = create_var(torch.from_numpy(l_features)).to(device).float()
    p_features = create_var(torch.from_numpy(p_features)).to(device).float()
    p_positions = [d[-2] for d in data]
    ligand_list = [d[-1] for d in data]

    return l_features, p_features, dm1_l_l, dm1_l_p, dm1_p_p, dm2_l_l, dm2_l_p, p_positions, ligand_list

def rotate_molecule(m, theta, axis):
    m = copy.deepcopy(m)
    rotation_matrix = get_rotation_matrix(axis, theta)
    c = m.GetConformers()[0]
    d = np.copy(c.GetPositions())
    for i in range(m.GetNumAtoms()):
        c.SetAtomPosition(i,np.copy(np.dot(rotation_matrix, d[i])))
    return m

def get_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def indices_to_smiles(indices, i_to_c):
    return ''.join([i_to_c[i] for i in indices])

