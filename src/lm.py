import torch
import torch.nn as nn
from graph import Graph
import layers


class LM(nn.Module):
    def __init__(self, vocab_sz, hparams, graph_predictor, logger, embd_weights=None):
        super(LM, self).__init__()
        self._hparams = hparams
        self.logger = logger
        self._V = vocab_sz
        self.G = graph_predictor
        self.layers = {}
        self.build()
        # if embedding weights are provided
        if embd_weights:
            self.layers['emb'].weight = nn.Parameter(embd_weights)
        self.modules = nn.ModuleList(list(self.layers.values()))    # has to register all modules, otherwise can't optimize 

    def build(self):
        self.drop = nn.Dropout(self._hparams['dropout'])
        emb_sz = self._hparams['embd_sz']
        hidden_sz = self._hparams['hidden_sz']
        # encoder
        self.layers['emb'] = nn.Embedding(self._V, emb_sz)
        cellClass = nn.GRUCell if self._hparams['compose_fn']=='GRUCell' else layers.ResLinear
        for l in range(1,self._hparams['L']+1):
            if l==1:
                cell = cellClass(emb_sz, hidden_sz)
            else:
                cell = cellClass(hidden_sz, hidden_sz)
            self.layers['encoder_%d'%l] = cell

        # decoder
        self.layers['decoder_rnn'] = nn.GRUCell(emb_sz,hidden_sz)
        self.layers['decoder_ff'] = nn.Linear(hidden_sz, self._V)
        if self._hparams['tie_weights']:
            assert emb_sz == hidden_sz, "Weight tying needs emb_sz==hidden_sz"
            self.layers['decoder_ff'].weights = self.layers['emb'].weights.t()
        

    def forward(self, x):
        '''
        args:
            x: [b,T] 
        return:
            out: Tensor [b, Dx(T-D),|V|], where 
                out[:,0:D,|V|]= decoder(<EOS>), 
                out[:, D:2D, |V|]=decoder(x0)
                ...
                out[:,(T-D-1)xD:(T-D)xD, |V|]=decoder(x_{T-D-2})
        '''
        x = x.type(torch.LongTensor)
        # compute graph affinity matrix
        # TODO: what exactly is the input to Graph? word idx or embedding?
        input_g = torch.unsqueeze(x,1).float()  # [b,1,T]
        G = self.G(input_g) #(b,L,T,T)
        T = x.shape[-1]
        b = x.shape[0]
        D = self._hparams['context_sz']
        
        # feature predictor -- encoder
        input_f = self.drop(self.layers['emb'](x)) # [b,T,embd_sz]
        embedded = input_f.clone()
        for l in range(1, self._hparams['L']+1):
            G_l = G[:,l-1,:,:]  #(b,T,T)
            # wgt_inputs = torch.zeros(b,T,input_f.shape[-1]) #(b,T,hidden)
            # for t in range(T):
            #     wgt_inputs[:,t,:] = torch.unsqueeze(G_l[:,:,t],1)@input_f.view(b,input_f.shape[-1])
            # Check: wgt_inputs[b,t,:] = sum_j{G[b,j,t])*f[b,j,:]}
            wgt_inputs = torch.transpose(G_l,1,2)@input_f           #(b,T,hidden)
            # if we use GRUCell, need to flatten inputs to [bxT, hidden] first...Linear layers does the flatten/unflatten for us
            self.logger.debug('l=%d, wgt_inputs.shape=%s, input_f.shape=%s' %(l, str(wgt_inputs.shape),str(input_f.shape)))
            if self._hparams['compose_fn']=='GRUCell':
                wgt_inputs = wgt_inputs.view(-1,wgt_inputs.shape[-1])
                input_f = input_f.view(-1,input_f.shape[-1])
            input_f = self.layers['encoder_%d'%l](wgt_inputs, input_f)      # [b,T,hidden]
            # if we use GRUCell, output is flattened as [bxT,hidden], so we need to fold it
            if self._hparams['compose_fn']=='GRUCell':
                input_f = input_f.view(b,T,-1)

        # decoder -- input_f: [b,T,hidden]
        out = []
        next_in = embedded[:,-1,:]      # input to decoder at t0: <EOS>
        for t in range(T-D):
            if t>0:
                next_in = embedded[:,t-1,:]     # input to decoder at t: x_{t-1}
            h0 = input_f[:,t,: ]                # init h0 of decoder at t: f_t
            xhat_t = []
            for d in range(self._hparams['context_sz']):
                h = self.layers['decoder_rnn'](next_in, h0)   # hidden state at d step: [b,hidden_sz]
                xhat_t_d = self.layers['decoder_ff'](h)         # output logits, [b,|V|]
                xhat_t.append(xhat_t_d)
                _, idx = xhat_t_d.max(-1)   # input to decoder at (d+1) step299  (b,)
                next_in = self.drop(self.layers['emb'](idx))     #[b,1,embd_sz]
            xhat = torch.stack(xhat_t,dim=1)    # [b,D,|V|]
            out.append(xhat)

        # should output a tensor of [b,Dx(T-D),|V|]
        return torch.cat(out,dim=1)
