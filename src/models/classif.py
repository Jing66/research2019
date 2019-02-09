import torch
import torch.nn as nn

from util import utils
from models.graph import Graph
import pdb

PAD=0

class Classifier(nn.Module):
    def __init__(self, v_sz, hparams, n_class, graph, logger=None, embd_weights=None):
        super(Classifier, self).__init__()
        self.graph = graph
        self._hparams = hparams
        self.n_class = n_class
        self.logger = logger
        self._V = v_sz
        self.build()
        if embd_weights:
            self.embedding.weight = nn.Parameter(embd_weights)
    

    def build(self):
        self.graph.freeze()
        hidden_sz = self._hparams['hidden_sz']    
        emb_sz = self._hparams['embd_sz']
        dense_sz = self._hparams['dense_sz']
        attn_heads = self._hparams['attn_heads']   

        self.embedding = nn.Embedding(self._V, emb_sz, padding_idx=PAD)
        self._drop = nn.Dropout(self._hparams['dropout'])
        
        rnn_type = getattr(nn, self._hparams['rnn_type'])
        self.rnn = rnn_type(hidden_sz,hidden_sz, num_layers=1, batch_first=True)
        self.linear_first = torch.nn.Linear(hidden_sz,dense_sz)
        self.linear_second = torch.nn.Linear(dense_sz,attn_heads)
        self.linear_final = nn.Linear(hidden_sz, self.n_class)      # binary classification
       # weighted sum of graph output: [n_layers,]
        n_layers = self._hparams['n_layers']
        self.mixture_wgt = nn.Parameter(data=torch.Tensor(n_layers), requires_grad=True)
        self.mixture_wgt.data.fill_(0.5)
        self.linear_cat1 = nn.Linear(emb_sz*2, hidden_sz,bias=False)
        self.linear_cat2 = nn.Linear(emb_sz*2, hidden_sz,bias=False)



    def graph_feature(self,x, mask):
        G, G_prod = self.graph.prop_connection(embedded, mask)
        n_layers = G.shape[1]
        m_g = self.mixture_wgt.view(1,n_layers,1,1)
        m_lambda = (1 - self.mixture_wgt).view(1,n_layers,1,1)
        M = m_g*G.detach() + w_lambda*G_prod.detach()           # [b,L,T,T]
        return torch.sum(M, dim=1)
        
    # https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py#L73
    def attn_softmax(self,x,dim=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    
    def fuse(self, H, M):
        '''
        H - [b, T, ndim]
        M - [b,T,T]
        return: H_ =  W1[H;HM] * sigmoid(W2[H;HM])
        '''
        weighted_input = H@M            # [b, T, ndim]
        cat_input = torch.cat((H, weighted_input),dim=1)        # [b,T, ndim*2]
        trans1 = self.linear_cat1(cat_input)
        trans2 = self.linear_cat2(cat_input)                    # [b, T, out_dim]
        return trans1 * F.sigmoid(trans2)

    def forward(self, x, lenghts, hidden_state):
        pdb.set_trace()
        b = x.shape[0]
        max_len = x.shape[1]
        embedded = self.embedding(x)            # [b,T,embd_sz]

        pad_mask = utils.get_mask_3d(x)
        M = self.graph_feature(x, pad_mask)         # [b, T, T] 
        
        next_in = self.fuse(embedded, M)
        outputs, hn  = self.rnn(next_in, hidden_state) 
        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x) 
        x = self.attn_softmax(x,1)
        attention = x.transpose(1,2)
        sentence_embeddings = attention@outputs 
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self._hparams['attn_heads']

        output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
        return output, hn


