import torch
import torch.nn as nn


class ResLinear(nn.Module):
    '''implement linear layer with residual connection'''
    def __init__(self, nb_in, nb_out):
        super(ResLinear,self).__init__()
        self.nb_in = nb_in
        self.nb_out = nb_out
        self.linear = nn.Linear(nb_in, nb_out)
        self.linear_res = nn.Linear(nb_in, nb_out)


    def forward(self, wgt_inputs, inputs):
        '''
            args: [*,nb_in]
            y = W1*wgt_inputs + W2*inputs
            return: [*,nb_out]
        '''
        xhat = self.linear(wgt_inputs)
        xres = self.linear(inputs)
        out = xhat + xres
        return out
