import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph(nn.Module):
    def __init__(self, max_len, hparams):
        super(Graph,self).__init__()
        self._T = max_len
        self._hparams = hparams
        self.layers = {}
        self.build()


    def build(self):
        for l in range(self._hparams['L']):
            # key and query CNN
            kernel_sz = self._hparams['kernel_sz'] # must be odd
            padding = int(0.5*kernel_sz) # s.t. conv output has dim T
            nb_filter = self._hparams['n_filter']
            k_conv = nn.Conv1d(in_channels=1, out_channels=nb_filter, kernel_size=kernel_sz, padding=padding)
            q_conv = nn.Conv1d(in_channels=1, out_channels=nb_filter, kernel_size=kernel_sz, padding=padding)
            linear_k = nn.Linear(self._T, k_conv.out_channels)
            linear_q = nn.Linear(self._T, q_conv.out_channels)

            b = nn.Parameters(torch.randn(self._T, self._T)) # bias term
            
            # collect layers
            self.layers['k_conv_%d'%l] = k_conv
            self.layers['q_conv_%d'%l] = q_conv
            self.layers['linear_k_%d'%l] = linear_k
            self.layers['linear_q_%d'%l] = linear_q
            self.layers['bias_%d'%l] = b



    def forward(self, x)
        ''' 
             x: [x_1...x_T] input, shape (b, d, T)
             return: G, shape (b, L, T, T)
        '''
        G_ = []
        for l in range(self._hparams['L']):
            ki = self.layers['k_conv_%d'%l](x)
            qi = self.layers['q_conv_%d'%l](x) # (b,d,T)
            kl = self.layers['linear_k_%d'%l](ki)
            ql = self.layers['linear_q_%d'%l](qi) #(b,d,T)
            # compute attention
            bias = self.layers['bias_%d'%l]
            G_l_unnorm = (F.relu(torch.transpose(kl)@ql) + bias)**2 # (b,T,T)
            G_l = G_l_unnorm/torch.sum(G_l_unnorm, dim=1, keepdim=True) #(b,T,T)
            G.append(G_l)
        G = torch.stack(G_, dim=1) # (b,L,T,T)
        return G

        
        

