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
            nb_filter_k = self._hparams['n_filter_k']
            nb_filter_q = self._hparams['n_filter_q']
            nb_lin_feat = self._hparams['linear_feat']
            k_conv = nn.Conv1d(in_channels=1, out_channels=nb_filter_k, kernel_size=kernel_sz, padding=padding)
            q_conv = nn.Conv1d(in_channels=1, out_channels=nb_filter_q, kernel_size=kernel_sz, padding=padding)
            linear_k = nn.Linear(k_conv.out_channels, nb_lin_feat)
            linear_q = nn.Linear(q_conv.out_channels, nb_lin_feat)

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
            qi = self.layers['q_conv_%d'%l](x) # (b, out_channel,T)
            # kl = self.layers['linear_k_%d'%l](ki)
            # ql = self.layers['linear_q_%d'%l](qi) #(b,d,T)
            # G_l_unnorm = (F.relu(torch.transpose(kl,1,2)@ql + bias))**2   # (b,T,T)
            kl = self.layers['linear_k_%d'%l](torch.transpose(ki,1,2))
            ql = self.layers['linear_q_%d'%l](torch.transpose(qi,1,2))      # (b,T,T)
            bias = self.layers['bias_%d'%l]
            # this computes: G_l[b,i,j] = [RELU(dot(kl[b,i,:],ql[b,j,:]+b)]^2
            G_l_unnorm = (F.relu(kl@torch.transpose(ql,1,2)+bias))**2       # (b,T,T)
            G_l = G_l_unnorm/torch.sum(G_l_unnorm, dim=1, keepdim=True)     #(b,T,T)
            G.append(G_l)
        G = torch.stack(G_, dim=1) # (b,L,T,T)
        return G

        
        

