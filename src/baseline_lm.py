import torch
import torch.nn as nn
import math
import pdb

PAD = 0


class BaseLM(nn.Module):
    '''Baseline neural language model'''

    def __init__(self, vocab_sz, hparams, logger, vocab_cutoff=None, embd_weights=None):
        super(BaseLM, self).__init__()
        # setup configs
        self._hparams = hparams
        self.logger = logger
        self._V = vocab_sz
        self.build(vocab_cutoff)
        # if embedding weights are provided
        if embd_weights:
            self._embedding.weight = nn.Parameter(embd_weights)
        self.apply(init_weights)
    
    

    def build(self, cutoffs):
        self._drop = nn.Dropout(self._hparams['dropout'])
        emb_sz = self._hparams['embd_sz']
        hidden_sz = self._hparams['hidden_sz']          # in message-passing, embedding size must equal GRU hidden size
        self._embedding = nn.Embedding(self._V, emb_sz, padding_idx=PAD)

        rnn_type = getattr(nn, self._hparams['rnn_type'])
        rnn_layers = self._hparams['rnn_layers']
        self._rnn = rnn_type(emb_sz,hidden_sz, num_layers=rnn_layers, batch_first=True)

        # Use Adaptive softmax instead of softmax(Linear) to save GPU memory
        _cluster_sz = 5
        div_val = math.log(hidden_sz/_cluster_sz, len(cutoffs)+1)
        div_val = min(4.0, math.floor(div_val*10)/10)
        self._decoder_remap = nn.AdaptiveLogSoftmaxWithLoss(hidden_sz, self._V, cutoffs, div_val)
        

    def forward(self, x, lengths, output_probs=False):
        is_cuda = x.is_cuda
        T = x.shape[-1]             # max length
        b = x.shape[0]
        inputs = self._drop(self._embedding(x))
        sos = inputs[torch.arange(b), lengths-1, :]
        inputs = torch.cat([torch.unsqueeze(sos,1),inputs],dim=1)  # [b,T+1, embd_sz]

        packed_input = nn.utils.rnn.pack_padded_sequence(inputs, lengths-1, batch_first=True)
        packed_output, last_state = self._rnn(packed_input)      
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)     # [b,T,hidden]
        output_ = output.view(b*(T-1), -1).contiguous()
        output = self._drop(output_)
        if output_probs:
            logprob_ = self._decoder_remap.log_prob(output)      # (b*T,)
            logprob = logprob_.view(b,T-1, -1)
            return logprob
        else:
            _, loss = self._decoder_remap(output,x[:,:-1].contiguous().view(-1,))
            return loss


        



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


