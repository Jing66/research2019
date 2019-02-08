import torch
import torch.nn as nn

from util import utils
from graph import Graph
from layers import SelfAttention
import pdb

PAD=1

class Classifier(nn.Module):
    def __init__(self, v_sz, hparams, graph, logger=Nonei, embd_weights=None):
        super(Classifier, self).__init__()
        self.graph = graph
        self._hparams = hparams
        self.logger = logger
        self._V = v_sz
        self.build()
        if embd_weights:
            self.embedding.weight = nn.Parameter(embd_weights)
    

    def build(self):
        self.graph.freeze()
        hidden_sz = self._hparams['hidden_sz']    
        emb_sz = self._hparams['embd_sz']
        self.embedding = nn.Embedding(self._V, emb_sz, padding_idx=PAD)
        self._drop = nn.Dropout(self._hparams['dropout'])
        
        rnn_type = getattr(nn, self._hparams['rnn_type'])
        rnn_layers = self._hparams['rnn_layers']
        self.rnn = rnn_type(emb_sz,hidden_sz, num_layers=rnn_layers, batch_first=True)
        # TODO: write attention module    
        self.attention = SelfAttention(hidden_sz)
        self.mlp = nn.Linear(hidden_sz, 2)      # binary classification
i       # weighted sum of graph output: [n_layers,]
        n_layers = self._hparams['n_layers']
        self.mixture_wgt = nn.Parameter(torch.data=torch.Tensor(n_layers)), requires_grad=True)
        self.mixture_wgt.data.fill_(0.5)


    def graph_feature(self,x, mask):
        with torch.no_grad:
            G, G_prod = self.graph.prop_connection(embedded, mask)
        n_layers = G.shape[1]
        m_g = self.mixture_wgt.view(1,n_layers,1,1)
        m_lambda = (1 - self.mixture_wgt).view(1,n_layers,1,1)
        M = m_g*G.detach() + w_lambda*G_prod.detach()           # [b,L,T,T]
        return M
        
    
    def forward(self, x, lenghts):
        embedded = self.embedding(x)

        pad_mask = utils.get_3d_mask(x)
        M = self.graph_feature(x, pad_mask) 


