import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparseattn
from models import layers
from util import utils
import pdb
EPSILON = 1e-9
SOFTMAX_MASK = -1e30

class Graph(nn.Module):
    def __init__(self, hparams, logger=None):
        super(Graph,self).__init__()
        self._hparams = hparams
        self.layers = {}
        self.logger = logger
        self.build()
        self.modules = nn.ModuleList(list(self.layers.values()))


    def build(self):
        for l in range(self._hparams['n_layers']):
            # key and query CNN
            kernel_sz = self._hparams['Graph']['kernel_sz'] # must be odd
            padding = int(0.5*kernel_sz) # s.t. conv output has dim T
            nb_filter_k = self._hparams['Graph']['n_filter_k']
            nb_filter_q = self._hparams['Graph']['n_filter_q']
            nb_lin_feat = self._hparams['Graph']['linear_feat']
            k_conv = nn.Conv1d(in_channels=self._hparams['embd_sz'], out_channels=nb_filter_k, kernel_size=kernel_sz, padding=padding)
            q_conv = nn.Conv1d(in_channels=self._hparams['embd_sz'], out_channels=nb_filter_q, kernel_size=kernel_sz, padding=padding)
            linear_k = nn.Linear(k_conv.out_channels, nb_lin_feat)
            linear_q = nn.Linear(q_conv.out_channels, nb_lin_feat)

            # collect layers
            self.layers['k_conv_%d'%l] = k_conv
            self.layers['q_conv_%d'%l] = q_conv
            self.layers['linear_k_%d'%l] = linear_k
            self.layers['linear_q_%d'%l] = linear_q
        # b = nn.Parameter(torch.zeros(1)) # bias term
        b = nn.Parameter(torch.zeros(self._hparams['n_layers']))
        self.graph_bias = b


    def forward(self, x, pad_mask):
        ''' 
             x: [x_1...x_T] input, shape (b, T, d)
             return: G, shape (b, L, T, T)
        '''
        # mask attention s.t it doesn't have access to future. [b,i:,i]=0
        subseq_mask = utils.get_subseq_mask(x)
        x = torch.transpose(x,1,2)          # [b,d,T]
        G_ = []
        for l in range(self._hparams['n_layers']):
            ki = self.layers['k_conv_%d'%l](x)
            qi = self.layers['q_conv_%d'%l](x)                  # (b, n_filters, T)
            kl = self.layers['linear_k_%d'%l](torch.transpose(ki,1,2))
            ql = self.layers['linear_q_%d'%l](torch.transpose(qi,1,2))      # (b,T,n_linear_feat)
            # bias = self.graph_bias
            bias = self.graph_bias[l]
            sparse_fn = self._hparams['Graph']['sparsity_fn']
            G_l_unnorm = kl@torch.transpose(ql,1,2)+bias
            mask = pad_mask & subseq_mask
            # this computes: G_l[b,i,j] = [fn(dot(kl[b,i,:],ql[b,j,:]+b)]^2
            # torch.sum(G_l, dim=1) should ==1
            if sparse_fn == 'leaky_relu' or sparse_fn == 'relu':
                sparse_fn = getattr(F, sparse_fn)
                G_l_unnorm = (sparse_fn(G_l_unnorm))**2       # (b,T,T)
                G_l_unnorm.masked_fill_(mask==0,0.0)
                # deal with extra sentence padding, add epsilon so that G doesn't divide by 0
                _tmp = torch.unsqueeze(pad_mask[:,:,0],1).expand(pad_mask.size())
                G_l_unnorm.masked_fill_(_tmp==0, EPSILON)
                Z = torch.sum(G_l_unnorm, dim=1, keepdim=True)                  #(b,T,T)
                G_l = G_l_unnorm/Z  
                G_l.masked_fill_(mask==0, 0.0)          # mask back the sentence padding
            elif sparse_fn == 'softmax':
                G_l_unnorm.masked_fill_(mask==0,SOFTMAX_MASK)
                G_l = F.softmax(G_l_unnorm, dim=1)
            elif sparse_fn == 'sparsemax':
                if not hasattr(self, "sparse_fn"):
                    self.sparse_fn = layers.Sparsemax(dim=1)
                # sparse_fn = torchsparseattn.SparsemaxFunction()
                G_l_unnorm.masked_fill_(mask==0,0.0)
                # T = G_l_unnorm.shape[1]
                # xflat = torch.transpose(G_l_unnorm,1,2).contiguous().view(-1,T)  # (b*T,T)
                # lengths = torch.sum(pad_mask.contiguous().view(-1,T),dim=1).detach()
                # y = sparse_fn(xflat).view(-1,T,T).contiguous()
                # G_l = torch.transpose(y,1,2)
                G_l = self.sparse_fn(G_l_unnorm)
            G_.append(G_l)
        G = torch.stack(G_, dim=1) # (b,L,T,T)
        return G

        
        

    def freeze(self):
        """Freeze graph network, used in downstreaming tasks"""
        for param in self.parameters():
            param.requires_grad = False


    def prop_connection(self, x, mask):
        '''propagate the connections among multiple layers
            returns:
            - G: graph affinity matrix
            - lambda: shape [b,L,T,T], lambda[:,l,:,:] = G[:,0,:,:]*...*G[:,l,:,:]
        '''
        G = self.forward(x, mask)           # [b,L,T,T]
        out = torch.cumprod(G, dim=1)       # [b,L,T,T]
        return G, out
