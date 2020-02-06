import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import utils

class RNN(nn.Module) :
    def __init__(self, args, n_char, i_to_c) :
        super(RNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.GRU = nn.GRU(input_size = args.n_feature, hidden_size = args.hidden_size, num_layers=args.n_layer)
        self.fc = nn.Linear(args.hidden_size, n_char)
        self.softmax = nn.Softmax(dim=2)
        self.embedding = nn.Embedding(n_char, args.n_feature)
        self.n_feature = args.n_feature
        self.hidden_size = args.hidden_size
        self.n_layer = args.n_layer
        self.i_to_c = i_to_c
        self.n_char = args.n_char
        self.start_codon = nn.Parameter(torch.zeros((args.n_feature)), requires_grad=True)
    
    def forward(self, x) :
        x_emb = self.embedding(x)                       #batch*len => batch*len*n_feature
        x_emb = x_emb.permute(1, 0, 2)                  #batch*len*n_feature => len*batch*n_feature
        start_codon = self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, x_emb.size(1), 1)
        input_data = torch.cat([start_codon, x_emb], 0)

        output, hidden = self.GRU(input_data)
        output = self.fc(output)                        #len*batch*n_feature => len*batch*n_char
        output = output.permute(1, 0, 2)                #len*batch*n_char => batch*len*n_char  
        return output

    def sampling(self, max_len) :
        result = ""
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
                if letter==self.n_char-1 :
                    break
                else :
                    codon = self.embedding(codon)
                    result+=self.i_to_c[letter]
            return result

class TransformerModel(nn.Module):
    def __init__(self, args, n_char, i_to_c):
        super(TransformerModel, self).__init__()
        self.pos_encoder = utils.PositionalEncoding(args.n_feature, args.dropout)
        encoder_layers = nn.TransformerEncoderLayer(args.n_feature, args.n_head, args.n_ff, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, args.n_layer)
        self.encoder = nn.Embedding(n_char, args.n_feature)
        self.n_input = args.n_feature
        self.decoder = nn.Linear(args.n_feature, n_char)
        self.src_mask = None
        self.softmax = nn.Softmax(dim=-1)
        self.start_codon = nn.Parameter(torch.zeros((args.n_feature)), requires_grad=True)
        self.n_char=n_char
        self.i_to_c=i_to_c

    def _generate_square_subsequent_mask(self, length) :
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True):
        src = src.permute(1, 0)                                                     #src : len*batch
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(0)+1:
                mask = self._generate_square_subsequent_mask(src.size(0)+1)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src)                                                     #len*batch => len*batch*n_feature
        startcodon=self.start_codon.unsqueeze(0).unsqueeze(0).repeat(1, src.size(1), 1)
        src=torch.cat([startcodon,src],0)                                           #len+1*batch*n_feature
        src=src*math.sqrt(self.n_input)
        src = self.pos_encoder(src)                                                 #len+1*batch*n_feature
        output = self.transformer_encoder(src, self.src_mask.to(device))            #src: len+1*batch*n_feature, mask: len+1*len+1
        output = self.decoder(output)
        output=output.permute(1,0,2)                                                #len+1* batch*n_char => batch*len+1*char
        return output

    def sampling(self, max_len):
        result = ""
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
                p_letter = self.softmax(output)
                output = torch.distributions.categorical.Categorical(p_letter)
                output = output.sample()
                letter = int(output[0][0])
                
                if letter==self.n_char-1 :
                    break
                else :
                    output = self.encoder(output)
                    codon = torch.cat([codon,output],0)
                    result+=self.i_to_c[letter]
            return result
        
