import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
import layers
import utils
import pdb

PAD = 0

class LM(nn.Module):
    def __init__(self, vocab_sz, hparams, graph_predictor, logger, vocab_cutoff=None,  embd_weights=None):
        super(LM, self).__init__()
        self._hparams = hparams
        self.logger = logger
        self._V = vocab_sz
        self.G = graph_predictor
        self.layers = {}
        self.build(vocab_cutoff)
        # if embedding weights are provided
        if embd_weights:
            self.layers['emb'].weight = nn.Parameter(embd_weights)
        self.modules = nn.ModuleList(list(self.layers.values()))    # has to register all modules, otherwise can't optimize 


    def build(self, cutoffs):
        '''Construct all layers
            cutoffs -- boundary for clusters in adaptive softmax'''
        pdb.set_trace()
        self.drop = nn.Dropout(self._hparams['dropout'])
        emb_sz = self._hparams['embd_sz']
        hidden_sz = emb_sz          # in message-passing, embedding size must equal GRU hidden size
        # encoder
        self.layers['emb'] = nn.Embedding(self._V, emb_sz, sparse=True, padding_idx=PAD)
        cellClass = nn.GRUCell if self._hparams['Feature']['compose_fn']=='GRUCell' else layers.ResLinear
        for l in range(1,self._hparams['n_layers']+1):
            if l==1:
                cell = cellClass(emb_sz, hidden_sz)
            else:
                cell = cellClass(hidden_sz, hidden_sz)
            self.layers['encoder_%d'%l] = cell

        # decoder
        p_ss = self._hparams['Feature']['SS_prob']
        if p_ss == 0.0:
            self.logger.info('Using teacher forcing in decoder')
            self.layers['decoder_rnn'] = nn.GRU(emb_sz,hidden_sz, num_layers=1, batch_first=True)
        else:
            self.layers['decoder_rnn'] = nn.GRUCell(emb_sz,hidden_sz)
        # Use Adaptive softmax instead of softmax(Linear) to save GPU memory
        self.layers['decoder_remap'] = nn.AdaptiveLogSoftmaxWithLoss(hidden_sz, self._V, cutoffs)

    def forward(self, x, lengths):
        '''
        args:
            x: [b,T] 
            lengths: [b,]
        return:
            out: Tensor [b, Dx(T-D),|V|], where 
                out[:,0:D,|V|]= decoder(<EOS>), 
                out[:, D:2D, |V|]=decoder(x0)
                ...
                out[:,(T-D-1)xD:(T-D)xD, |V|]=decoder(x_{T-D-2})
        '''
        is_cuda = x.is_cuda
        T = x.shape[-1]             # max length
        b = x.shape[0]
        D = self._hparams['Feature']['context_sz']
        mask = utils.get_mask_3d(x)       # (b,T,T)
        if is_cuda:
            mask = mask.cuda()
        # feature predictor -- encoder
        input_f = self.drop(self.layers['emb'](x)) # [b,T,embd_sz]
        embedded = input_f.clone()
        # compute graph affinity matrix
        input_g = torch.transpose(embedded,1,2)         # [b,embd_sz, T]
        G = self.G(input_g, mask)                #(b,L,T,T)

        for l in range(1, self._hparams['n_layers']+1):
            G_l = G[:,l-1,:,:]  #(b,T,T)
            # wgt_inputs[b,t,:] = sum_j{G[b,j,t])*f[b,j,:]}
            wgt_inputs = torch.transpose(G_l,1,2)@input_f           #(b,T,hidden)
            # if we use GRUCell, need to flatten inputs to [bxT, hidden] first...Linear layers does the flatten/unflatten for us
            self.logger.debug('l=%d, wgt_inputs.shape=%s, input_f.shape=%s' %(l, str(wgt_inputs.shape),str(input_f.shape)))
            if self._hparams['Feature']['compose_fn']=='GRUCell':
                wgt_inputs = wgt_inputs.view(-1,wgt_inputs.shape[-1])
                input_f = input_f.view(-1,input_f.shape[-1])
            input_f = self.layers['encoder_%d'%l](wgt_inputs, input_f)      # [b,T,hidden]
            # if we use GRUCell, output is flattened as [bxT,hidden], so we need to fold it
            if self._hparams['Feature']['compose_fn']=='GRUCell':
                input_f = input_f.view(b,T,-1)

        # decoder -- input_f: [b,T,hidden]
        logprobs = []
        next_in = embedded[torch.arange(b),lengths-1,:]      # input to decoder at t0: <EOS>
        n_padding= 0
        # scheduled-sampling
        p_ss = self._hparams['Feature']['SS_prob']      # with prob p_ss, use the predicted word as input

        def _select_by_length( timestep, lens):
            '''select only non-padding batch into RNN. return number of samples excluded
            '''
            #zeros = torch.zeros(lens.shape).type(torch.LongTensor)
            #if is_cuda:
            #    zeros = zeros.cuda()
            #select = torch.where( lens>timestep, zeros, lens)
            #n_padding = torch.sum((select!=0))
            n_padding = torch.sum(lens<=timestep)
            return n_padding

        for t in range(T-D+1):
            h0_t = input_f[:,t,: ]                # init h0 of decoder at t: f_t (b,hidden_sz)
            next_hidden = h0_t

            # pdb.set_trace()
            # CASE teacher forcing, use packed_padded_seq to speed up
            if p_ss == 0.0:
                if t>0:
                    inputs = embedded[:, t-1:t-1+D,: ]   # inputs: [b,D,hidden]
                else:
                    inputs = torch.cat((torch.unsqueeze(next_in,1), embedded[:,:D-1,: ]),1)
                packed_input = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
                packed_output, _ = self.layers['decoder_rnn'](packed_input, torch.unsqueeze(h0_t,0).contiguous())      
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)     # [b,D,hidden]
                logprob = self.layers['decoder_remap'](output.view(b*D, -1)).view(b,D,-1)          #[b,D, |V|]

            # CASE use scheduled sampling
            else:
                if t>0:
                    next_ = embedded[:,t-1,:]     # input to decoder at t: x_{t-1}
                    n_padding = _select_by_length(t, lengths)       # next_in: [b_, embd_sz]
                    next_in, next_hidden = next_[:(b-n_padding)], h0_t[:(b-n_padding)]
                xhat_t = []
                for d in range(D):
                    h = self.layers['decoder_rnn'](next_in, next_hidden)   # hidden state at d step: [b_,hidden_sz]
                    xhat_t_d_ = self.layers['decoder_remap'].log_prob(h)         # output logits: [b,|V|]
                    xhat_t_d = F.pad(xhat_t_d_, ( 0,0,0,n_padding))
                    xhat_t.append(xhat_t_d)
                    _, idx = xhat_t_d.max(-1)   # input to decoder at (d+1) step  (b_,)
                    
                    if torch.randn(1)[0].item() > p_ss:
                        next_ = embedded[:,t+d,:]
                    else:
                        next_ = self.drop(self.layers['emb'](idx))     # [b_,embd_sz]
                    
                    if t>0 :
                        n_padding = _select_by_length(t+d+1, lengths)
                        next_in = next_[: (b - n_padding)]
                        next_hidden = h[:(b-n_padding)]
                    else:
                        next_in, next_hidden = next_, h
            
                logprob = torch.stack(xhat_t,dim=1)    # [b,D,|V|]
            logprobs.append(logprob)
        # should output a tensor of [b,Dx(T-D+1),|V|]
        return torch.cat(logprobs,dim=1)
