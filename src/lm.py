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
        cellClass = nn.GRUCell if self._hparams['compose_fn']=='GRUCell' else layers.ResLinear
        for l in range(1,self._hparams['L']+1):
            if l==1:
                cell = cellClass(embd_sz, hidden_sz)
            else:
                cell = cellClass(hidden_sz, hidden_sz)
            self.layers['encoder_%d'%l] = cell

        # decoder
        self.layers['decoder_rnn'] = nn.GRUCell(emb_sz,hidden_sz)
        self.layers['decoder_ff'] = nn.Linear(hidden_sz, self._V)
        if self._hparams['tie_weights']:
            assert emb_sz == hidden_sz, "Weight tying needs emb_sz==hidden_sz"
            self.layers['decoder_ff'].weights = self.layers['emb'].weights.t()
        

    def forward(self, x, lens):
        '''
        args:
            x: [b,T] sorted by length in a decreasing order
            lens: [b,]
        '''
        # compute graph affinity matrix
        input_g = torch.unsqueeze(x,1)  # [b,1,T]
        G = self.G(input_g) #(b,L,T,T)
        T = G.shape(-1)
        b = G.shape[0]
        
        # feature predictor -- encoder
        input_f = self.drop(self.layers['emb'](x) # [b,T,embd_sz]
        embedded = input_f.clone()
        for l in range(1, self._hparams['L']+1):
            G_l = G[:,l-1,:,:]  #(b,T,T)
            # wgt_inputs = torch.zeros(b,T,input_f.shape[-1]) #(b,T,hidden)
            # for t in range(T):
            #     wgt_inputs[:,t,:] = torch.unsqueeze(G_l[:,:,t],1)@input_f.view(b,input_f.shape[-1])
            # Check: wgt_inputs[b,t,:] = sum_j{G[b,j,t])*f[b,j,:]}
            wgt_inputs = torch.transpose(G_l,1,2)@input_f           #(b,T,hidden)
            # if we use GRUCell, need to flatten inputs to [bxT, hidden] first...Linear layers does the flatten/unflatten for us
            if self._hparams['compose_fn']=='GRUCell':
                wgt_inputs = wgt_inputs.view(-1,wgt_inputs.shape[-1])
                input_f = input_f.view(-1,input_f.shape[-1])
            input_f = self.layers['encoder_%d'%d](wgt_inputs, input_f)      # [b,T,hidden]
            # if we use GRUCell, output is flattened as [bxT,hidden], so we need to fold it
            if self._hparams['compose_fn']=='GRUCell'::
                input_f = input_f.view(b,T,-1)

        # decoder -- input_f: [b,T,hidden]
        out = []
        for t in range(T):
            xhat_t = []
            for d in range(self._hparams['context_sz']):
                xhat_t_d = self.layers['decoder_rnn'](embedded[:,t,:], input_f[:,t,:])
                xhat_t.append(self.layers['decodre_ff'])(xhat_t_d)
            xhat = torch.stack(xhat_t,dim=1)    # [b,D,|V|]
            out.append(xhat)

        # should output a tensor of [b,DxT,|V|]
        return torch.stack(out,dim=1)
