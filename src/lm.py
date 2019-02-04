import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
import layers
import utils
import pdb

PAD = 0

class LM(nn.Module):
    def __init__(self, vocab_sz, hparams, graph_predictor, logger, 
                            vocab_cutoff=None, embd_weights=None):
        '''
        Args:
            - graph_predictor: class Graph()
            - vocab_cutoff: list of elements as boundaries of clusters for the last layer, used in adaptiveLogSoftmax
            - embd_weights: np.ndarray. use pretrained embeddings
        '''
        super(LM, self).__init__()
        # setup configs
        self._hparams = hparams
        self.logger = logger
        self._V = vocab_sz
        self.G = graph_predictor
        self.layers = {}
        self.build(vocab_cutoff)
        # if embedding weights are provided
        if embd_weights:
            self.layers['emb'].weight = nn.Parameter(embd_weights)
        self.blocks = nn.ModuleList(list(self.layers.values()))
        self.apply(init_weights)


    def build(self, cutoffs):
        '''Construct all layers
            cutoffs -- boundary for clusters in adaptive softmax'''
        self.drop = nn.Dropout(self._hparams['dropout'])
        emb_sz = self._hparams['embd_sz']
        hidden_sz = emb_sz          # in message-passing, embedding size must equal GRU hidden size
        # encoder
        self.layers['emb'] = nn.Embedding(self._V, emb_sz, padding_idx=PAD)
        cellClass = nn.GRUCell if self._hparams['Feature']['compose_fn']=='GRUCell' else layers.ResLinear
        for l in range(1,self._hparams['n_layers']+1):
            cell = cellClass(hidden_sz, hidden_sz)
            self.layers['encoder_%d'%l] = cell

        # decoder
        p_ss = self._hparams['Feature']['SS_prob']
        self.logger.info("With prob %d, use decoded output as next input" %p_ss)
        self.layers['decoder_rnn'] = nn.GRU(emb_sz,hidden_sz, num_layers=1, batch_first=True)

        # Use Adaptive softmax instead of softmax(Linear) to save GPU memory
        _cluster_sz = 5
        div_val = math.log(hidden_sz/_cluster_sz, len(cutoffs)+1)
        div_val = min(4.0, math.floor(div_val*10)/10)
        self.layers['decoder_remap'] = nn.AdaptiveLogSoftmaxWithLoss(hidden_sz, self._V, cutoffs, div_val)


    def attn_fn(self, x):
        '''return attention weights from graph'''
        if len(x.shape)==1:
            x = torch.unsqueeze(x,0)
        embd = self.layers['emb'](x)
        b = 1
        T = x.shape[-1]
        mask = utils.get_mask_3d(x)       # (b,T,T)
        g_matrix = self.G(embd, mask)      # [b,L,T,T]
        return g_matrix.detach()



    def forward(self, x, lengths, output_probs=False):
        '''
        Args:
            - x: [b,T] 
            - output_probs: bool. if True, model output log probs [b,(T-1)*D, |V|] instead of averaged loss (scalar). NOTE: output loss is much more efficient in both speed and memory, due to the use of `adaptiveLogSoftmax`
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
        # feature predictor -- encoder
        input_f = self.drop(self.layers['emb'](x)) # [b,T,embd_sz]
        mask = utils.get_mask_3d(x)       # (b,T,T)
        if is_cuda:
            mask = mask.cuda()
        embedded = input_f.clone()
        # compute graph affinity matrix
        G = self.G(embedded, mask)                #(b,L,T,T)
        # pdb.set_trace()

        for l in range(1, self._hparams['n_layers']+1):
            G_l = G[:,l-1,:,:]  #(b,T,T)
            # wgt_inputs[b,t,:] = sum_j{G[b,j,t])*f[b,j,:]}
            wgt_inputs = torch.transpose(G_l,1,2)@input_f           #(b,T,hidden)
            # if we use GRUCell, need to flatten inputs to [bxT, hidden] first...Linear layers does the flatten/unflatten for us
            if self._hparams['Feature']['compose_fn']=='GRUCell':
                wgt_inputs = wgt_inputs.view(-1,wgt_inputs.shape[-1])
                input_f = input_f.view(-1,input_f.shape[-1])
            input_f = self.layers['encoder_%d'%l](wgt_inputs, input_f)      # [b,T,hidden]
            # if we use GRUCell, output is flattened as [bxT,hidden], so we need to fold it
            if self._hparams['Feature']['compose_fn']=='GRUCell':
                input_f = input_f.view(b,T,-1)

        # ----------------------- decoder -- input_f: [b,T,hidden] -----------------
        logprobs = []
        next_in = embedded[torch.arange(b),lengths-1,:]      # input to decoder at t0: <EOS>
        n_padding= 0
        # scheduled-sampling
        p_ss = self._hparams['Feature']['SS_prob']      # with prob p_ss, use the predicted word as input

        def _select_by_length( timestep, lens):
            '''select only non-padding batch into RNN. return number of samples excluded
            '''
            n_padding = torch.sum(lens<=timestep)
            return n_padding.detach().item()

        for t in range(T-D+1):
            h0_t = input_f[:,t,: ]                # init h0 of decoder at t: f_t (b,hidden_sz)

            # ------ CASE teacher forcing, use packed_padded_seq to speed up
            if p_ss == 0.0:
                if t>0:
                    inputs = embedded[:, t-1:t-1+D,: ]   # inputs: [b,D,hidden]
                else:
                    inputs = torch.cat((torch.unsqueeze(next_in,1), embedded[:,:D-1,: ]),1)
                n_padding = _select_by_length(t,lengths)
                inputs, h0_t = inputs[:(b-n_padding)], h0_t[:(b-n_padding)]
                _tmp = torch.tensor([D]).type(torch.LongTensor)
                if is_cuda:
                    _tmp = _tmp.cuda()
                lens = torch.min(lengths[:(b-n_padding)]-t, _tmp.expand(inputs.shape[0])).detach()
                packed_input = nn.utils.rnn.pack_padded_sequence(inputs, lens, batch_first=True)
                packed_output, _ = self.layers['decoder_rnn'](packed_input, torch.unsqueeze(h0_t,0).contiguous())      
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)     # [b,D,hidden]
                output = output.transpose(0,1).contiguous().view((b-n_padding)*D,-1).contiguous()
                if output_probs:
                    logprob = self.layers['decoder_remap'].log_prob(output).view((b-n_padding),D,-1)          #[b,D, |V|]
                    logprob = F.pad(logprob, (0,0,0,0,0,n_padding))
                    logprobs.append(logprob)
                else:
                    _, loss = self.layers['decoder_remap'](output, x[:(b-n_padding),t:t+D].contiguous().view(-1))
                    logprobs.append(loss) 

            # ----- CASE use scheduled sampling
            else:
                next_hidden = h0_t
                if t>0:
                    next_ = embedded[:,t-1,:]     # input to decoder at t: x_{t-1}
                    n_padding = _select_by_length(t, lengths)       # next_in: [b_, embd_sz]
                    next_in, next_hidden = next_[:(b-n_padding)], h0_t[:(b-n_padding)]
                next_hidden = torch.unsqueeze(next_hidden,0).contiguous()
                xhat_t = []
                for d in range(D):
                    next_in = torch.unsqueeze(next_in,1).contiguous()
                    _,h = self.layers['decoder_rnn'](next_in, next_hidden)   # hidden state at d step: [1, b_, hidden_sz]
                    xhat_t_d_ = self.layers['decoder_remap'].log_prob(torch.squeeze(h,0))         # output logits: [b,|V|]
                    xhat_t_d = F.pad(xhat_t_d_, ( 0,0,0,n_padding))
                    xhat_t.append(xhat_t_d)
                    _, idx = xhat_t_d.max(-1)   # input to decoder at (d+1) step  (b_,)
                    if torch.rand(1).item() > p_ss:
                        next_ = embedded[:,t+d,:]
                    else:
                        next_ = self.drop(self.layers['emb'](idx))     # [b_,embd_sz]
                    
                    n_padding = _select_by_length(t+d+1, lengths)
                    next_in = next_[: (b - n_padding)]
                    next_hidden = h[:,:(b-n_padding),:]
            
                logprob = torch.stack(xhat_t,dim=1)    # [b,D,|V|]
                logprobs.append(logprob)
        if output_probs:
            # output a tensor of [b,Dx(T-D+1),|V|]
            return torch.cat(logprobs,dim=1)
        else:
            return sum(logprobs)/len(logprobs)





def init_weights(module):
    '''weight initialization for the graph/LM model'''
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None: 
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.GRU):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
    elif isinstance(module, nn.GRUCell):
        nn.init.xavier_uniform_(module.weight_ih.data)
        nn.init.orthogonal_(module.weight_hh.data)
        nn.init.constant_(module.bias_ih.data, 0.0)
        nn.init.constant_(module.bias_hh.data, 0.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight.data) 

