import torch
import torch.nn as nn
from graph import Graph


class LM(nn.Module):
    def __init__(self, vocab_sz, hparams, graph_predictor, embd_weights=None):
        super(FeatureLM, self).__init__()
        self._hparams = hparams
        self._V = vocab_sz
        self.G = graph_predictor
        self.layers = {}
        self.build()
        # if embedding weights are provided
        if embd_weights:
            self.layers['emb'].weight = nn.Parameter(embd_weights)


    def build(self):
        self.drop = nn.Dropout(self._hparams['dropout'])
        emb_sz = self._hparams['embd_sz']
        hidden_sz = self._hparams['hidden_sz']
        # encoder
        self.layers['emb'] = nn.Embedding(self._V, emb_sz)
        for l in range(1,self._hparams['L']+1):
            if l==1:
                cell = nn.GRUCell(embd_sz, hidden_sz)
            else:
                cell = nn.GRUCell(hidden_sz, hidden_sz)
            self.layers['encoder_%d'%l] = cell

        # decoder
        self.layers['decoder_rnn'] = nn.GRU(hidden_sz, hidden_sz,1, batch_first=True)
        self.layers['decoder_ff'] = nn.Linear(hidden_sz, self._V)
        if self._hparams['tie_weights']:
            self.layers['decoder_ff'].weight = self.layers['emb'].weights
        

    def forward(self, x, lens):
        '''
        args:
            x: [b,T] sorted by length in a decreasing order
            lens: [b,]
        '''
        # compute graph affinity matrix
        input_g = torch.unsqueeze(x,1)  # [b,1,T]
        G = self.G(input_g) #(b,L,T,T)
        
        # feature predictor -- encoder
        input_f  = self.drop(self.layers['emb'](x) # [b,T, embd_sz]
        for l in range(1, self._hparams['L']+1):
            input_f_ts=[]
            # eq(2)
            weights = G[:,l-1,:,:] * input_f # [b,T,T]
            weighted = torch.sum(weights,1) # [b,T, T]
            for t in range(x.shape[1]):
                input_f_t = self.layers['encoder_%d'%l](weighted[:,:,t], input_f[:,t,:]) #[b,T]
                input_f_ts.append(input_f_t)
            input_f = torch.cat(input_f_ts,2) #[b,T,T]            
        
        # decoder -- obj func
        dec = self.layers['decoder_rnn'](input_f,input_f)
        out = self.layers['decoder_ff'](dec)
