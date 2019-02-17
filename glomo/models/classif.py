import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util import utils
from models.graph import Graph
import pdb

PAD=0
SOFTMAX_MASK = -1e30

class Classifier(nn.Module):
    def __init__(self, v_sz, hparams, n_class, graph, logger=None, embd_weights=None):
        super(Classifier, self).__init__()
        self.graph = graph
        self._hparams = hparams
        self.n_class = n_class
        self.logger = logger
        self._V = v_sz
        if embd_weights is not None: 
            self.logger.info("Model using pretrained embeddings, dim=%d"%embd_weights.shape[1])
            self._hparams['embd_sz'] =embd_weights.shape[1]
        self.build()
        self.graph.freeze()
        if embd_weights is not None:
            self.embedding.weight = nn.Parameter(embd_weights)
    

    def build(self):
        hidden_sz = self._hparams['Feature']['hidden_sz']    
        emb_sz = self._hparams['embd_sz']
        dense_sz = self._hparams['Feature']['dense_sz']
        attn_heads = self._hparams['Feature']['attn_heads']   

        self.embedding = nn.Embedding(self._V, emb_sz, padding_idx=PAD)
        self._drop = nn.Dropout(self._hparams['dropout'])
        
        rnn_type = getattr(nn, self._hparams['Feature']['rnn_type'])
        self.rnn = rnn_type(hidden_sz,hidden_sz, num_layers=1, batch_first=True)
        # self.rnn = rnn_type(emb_sz, hidden_sz, num_layers=1, batch_first=True)
        self.linear_first = torch.nn.Linear(hidden_sz,dense_sz)
        self.linear_second = torch.nn.Linear(dense_sz,attn_heads)
        self.linear_final = nn.Linear(hidden_sz, self.n_class)      # binary classification
        # weighted sum of graph output: [n_layers,]
        n_layers = self._hparams['n_layers']
        self.mixture_wgt = nn.Parameter(data=torch.Tensor(n_layers), requires_grad=True)
        self.mixture_wgt.data.fill_(0.5)
        self.linear_cat1 = nn.Linear(emb_sz*2, hidden_sz,bias=False)
        self.linear_cat2 = nn.Linear(emb_sz*2, hidden_sz,bias=False)



    def graph_feature(self,embeddings, mask):
        G, G_prod = self.graph.prop_connection(embeddings, mask)
        n_layers = G.shape[1]
        m_g = self.mixture_wgt.view(1,n_layers,1,1)
        m_lambda = (1 - self.mixture_wgt).view(1,n_layers,1,1)
        M = m_g*G.detach() + m_lambda*G_prod.detach()           # [b,L,T,T]
        return torch.sum(M, dim=1)
        
    
    def fuse(self, H, M):
        '''
        H - [b, T, ndim]
        M - [b,T,T]
        return: H_ =  W1[H;HM] * sigmoid(W2[H;HM])
        '''
        weighted_input = torch.transpose(M,1,2)@H           # [b, T, ndim]
        cat_input = torch.cat((H, weighted_input),dim=2)        # [b,T, ndim*2]
        trans1 = self.linear_cat1(cat_input)
        trans2 = self.linear_cat2(cat_input)                    # [b, T, out_dim]
        return trans1 * torch.sigmoid(trans2)

    def forward(self, x, lengths, hidden_state):
        '''
        Args:
            - x: (bxT)
            - lengths: (b,)
        Return:
            - output: (b x n_class) unnormliazed logits
            - hidden_state: tuple of rnn hidden states
        '''
        # pdb.set_trace()
        b = x.shape[0]
        hidden_state = utils.slice_(hidden_state,b)
        max_len = x.shape[1]
        attn_heads = self._hparams['Feature']['attn_heads'] 

        embedded = self.embedding(x)            # [b,T,embd_sz]

        pad_mask = utils.get_mask_3d(x)
        mask = utils.get_mask_2d(lengths)          # [b,T]
        mask_attn = torch.unsqueeze(mask,-1).expand(b, max_len, attn_heads)

        M = self.graph_feature(embedded, pad_mask)         # [b, T, T]
        next_in = self.fuse(embedded, M)                    # [b,T, hidden_sz]
        # next_in = embedded
        outputs, hn  = self.rnn(next_in, hidden_state) 
        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)                           # [b, T, #head] 
        x = x.masked_fill_(mask_attn==0, SOFTMAX_MASK)
        x = F.softmax(x, dim=1)
        attention = x.transpose(1,2)                        # [b, #head, T]
        sentence_embeddings = attention@outputs             # [b, #head, hidden_sz]
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/attn_heads
        
        output = self.linear_final(avg_sentence_embeddings)
        return output, hn, attention

    @classmethod
    def repackage(cls,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == torch.Tensor or type(h)==Variable:
            return Variable(h.detach().data)
        else:
            return tuple(cls.repackage(v) for v in h)


    def l2_matrix_norm(self,m):
        '''
        Frobenius norm calculation:
            m: {Variable} ||AAT - I||
        '''
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5)


