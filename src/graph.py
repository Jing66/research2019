import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import pdb
EPSILON = 1e-4


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
            self.logger.debug('kernel_size=%d, padding=%d'%(kernel_sz, padding))
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
        # b = nn.Parameters(torch.randn(self._T, self._T)) # bias term
        b = nn.Parameter(torch.zeros(self._hparams['n_layers']))
        self.graph_bias = b


    def forward(self, x, mask):
        ''' 
             x: [x_1...x_T] input, shape (b, T, d)
             return: G, shape (b, L, T, T)
        '''
        x = torch.transpose(x,1,2)          # [b,d,T]
        G_ = []
        for l in range(self._hparams['n_layers']):
            # pdb.set_trace()
            ki = self.layers['k_conv_%d'%l](x)
            qi = self.layers['q_conv_%d'%l](x)                  # (b, n_filters, T)
            kl = self.layers['linear_k_%d'%l](torch.transpose(ki,1,2))
            ql = self.layers['linear_q_%d'%l](torch.transpose(qi,1,2))      # (b,T,n_linear_feat)
            bias = self.graph_bias[l]
            # this computes: G_l[b,i,j] = [RELU(dot(kl[b,i,:],ql[b,j,:]+b)]^2
            sparse_fn = getattr(F, self._hparams['Graph']['sparsity_fn'])
            G_l_unnorm = (sparse_fn(kl@torch.transpose(ql,1,2)+bias))**2       # (b,T,T)
            G_l_masked = G_l_unnorm * mask
            Z = torch.sum(G_l_masked, dim=1, keepdim=True)                  #(b,T,T)
            G_l = G_l_masked/(Z+EPSILON)                    # Z might be zero since RELU sets  all neg. values to 0
            G_.append(G_l)
        G = torch.stack(G_, dim=1) # (b,L,T,T)
        return G

        
        

    def freeze(self):
        """Freeze graph network, used in downstreaming tasks"""
        for param in self.parameters():
            param.requires_grad = False


    def prop_connection(self, x, mask):
        '''propagate the connections among multiple layers
            return lambda: shape [b,L,T,T], lambda[:,l,:,:] = G[:,0,:,:]*...*G[:,l,:,:]
        '''
        G = self.forward(x, mask)           # [b,L,T,T]
        out = torch.cumprod(G, dim=1)       # [b,L,T,T]
        return out
