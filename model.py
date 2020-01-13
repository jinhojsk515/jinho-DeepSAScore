import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class RNN(nn.Module) :
    def __init__(self, n_feature, hidden_size, n_char, n_layer, i_to_c) :
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.GRU = nn.GRU(input_size = n_feature, hidden_size = hidden_size, num_layers=n_layer)
        self.fc = nn.Linear(hidden_size, n_char)
        self.softmax = nn.Softmax(dim=2)
        self.embedding = nn.Embedding(n_char, n_feature)
        self.n_feature = n_feature
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.i_to_c = i_to_c
        self.n_char = n_char
        self.start_codon = nn.Parameter(torch.zeros((n_feature)), requires_grad=True)
    
    def forward(self, x) :
        x_emb = self.embedding(x)                       #batch*len => batch*len*n_feature
        x_emb = x_emb.permute(1, 0, 2)                  #batch*len*n_feature => len*batch*n_feature
        start_codon = self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, x_emb.size(1), 1)
        input_data = torch.cat([start_codon, x_emb], 0)

        output, hidden = self.GRU(input_data)
        output = self.fc(output)                        #len*batch*n_feature => len*batch*n_char
        output = output.permute(1, 0, 2)                #len*batch*n_char => batch*len*n_char
        p_char = self.softmax(output)
              
        return output, p_char


    def sampling(self, max_len) :
        result = ""
        p=0
        with torch.no_grad() :
            codon = self.start_codon.unsqueeze(0).unsqueeze(0)
            hidden = torch.zeros(self.n_layer, 1, self.hidden_size).to(codon.device)
            for _ in range(max_len) :
                codon, hidden = self.GRU(codon, hidden)
                codon = self.fc(codon)
                p_letter = self.softmax(codon)
                codon = torch.distributions.categorical.Categorical(p_letter)
                codon = codon.sample()
                letter = int(codon[0][0])
                p+=float(torch.log(p_letter[0][0][letter]))
                if letter==self.n_char-1 :
                    break
                else :
                    codon = self.embedding(codon)
                    result+=self.i_to_c[letter]
            return result, p



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        print ('dropout',dropout)
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


class TransformerModel(nn.Module):
    def __init__(self, n_char, n_input, n_head, n_hidden, n_layers, batch_size, n_ff, i_to_c, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(n_input, dropout)
        encoder_layers = nn.TransformerEncoderLayer(n_input, n_head, n_ff, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_char, n_input)
        self.n_input = n_input
        self.decoder = nn.Linear(n_input, n_char)
        self.src_mask = None
        self.softmax = nn.Softmax(dim=2)
        self.start_codon = nn.Parameter(torch.zeros((n_input)), requires_grad=True)
        self.n_char=n_char
        self.i_to_c=i_to_c

    def _generate_square_subsequent_mask(self, length) :
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True):
        src = src.permute(1, 0)                                 #src : len * batch
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(0)+1:
                mask = self._generate_square_subsequent_mask(src.size(0)+1).to(device)
                self.src_mask = mask
            else:
                self.src_mask = None
        src = self.encoder(src)                                 #len*batch => len*batch*n_feature
        startcodon=self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, src.size(1), 1)
        src=torch.cat([startcodon,src],0)                       #len+1*batch*in_feature
        src=src*math.sqrt(self.n_input)
        src = self.pos_encoder(src)                             #len+1*batch*n_feature
        output = self.transformer_encoder(src, self.src_mask)   #src: len+1*batch*n_feature, mask: len+1*len+1
        output = self.decoder(output)
        output=output.permute(1,0,2) #batch*len+1*char
        return F.log_softmax(output, dim=-1), output

    def sampling(self, max_len):
    
        result = ""
        p=0
        with torch.no_grad() :
            codon = self.start_codon.unsqueeze(0).unsqueeze(0)
            for _ in range(max_len) :
                device=codon.device
                mask = self._generate_square_subsequent_mask(codon.size(0)).to(device)
                self.src_mask=mask

                output = codon * math.sqrt(self.n_input)
                output = self.pos_encoder(output)
                output = self.transformer_encoder(output,self.src_mask)
                output = output[-1:][:][:]
                output = self.decoder(output)

            #using max, not categorical
                #p_letter = self.softmax(output)
                #_, output = torch.max(p_letter,-1)
                #letter = int(output[0][0])

                p_letter = self.softmax(output)
                output = torch.distributions.categorical.Categorical(p_letter)
                output = output.sample()
                letter = int(output[0][0])
                p+=float(torch.log(p_letter[0][0][letter]))
                
                if letter==self.n_char-1 :
                    break
                else :
                    output = self.encoder(output)
                    codon = torch.cat([codon,output],0)
                    result+=self.i_to_c[letter]
            return result, p
        
