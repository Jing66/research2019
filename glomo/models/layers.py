import torch
import torch.nn as nn
from torch.autograd import Function

import pdb

import sys
sys.path.append('/home/ml/jliu164/GLoMo/glomo')
from util import utils

def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0, mask=None):
    """
    Sparsemax building block: compute the threshold
    Parameters:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """
    if mask is not None:
        input.masked_fill_(mask==0,-1e9)
    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, input, dim=0,mask=None):
        """
        sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim, mask=mask)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input,mask=None):
        return sparsemax(input, self.dim,mask)



class ResLinear(nn.Module):
    '''implement linear layer with residual connection'''
    def __init__(self, nb_in, nb_out):
        super(ResLinear,self).__init__()
        self.nb_in = nb_in
        self.nb_out = nb_out
        self.linear = nn.Linear(nb_in, nb_out)


    def forward(self, wgt_inputs, inputs):
        '''
            args: [*,nb_in]
            y = W1*wgt_inputs + W2*inputs
            return: [*,nb_out]
        '''
        xhat = self.linear(wgt_inputs)
        out = xhat + inputs
        return out



def test_sparsemax():
    fn = Sparsemax(dim=1)
    fn2 = Sparsemax(dim=0)
    torch.manual_seed(2)
    inputs = torch.randn(3,5,5)
    pad_mask = torch.tensor([[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]],
                            [[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[0,0,0,0,0]],
                           [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]], dtype=torch.uint8)
    subseq_mask =torch.triu(torch.ones((5,5), dtype=torch.uint8), diagonal=1).unsqueeze(0).expand(3, -1, -1)
    mask = subseq_mask & pad_mask
    inputs.masked_fill_(mask==0,0.0)
    output = fn(inputs,subseq_mask)
    print("output",output) 

# test sparsemax
if __name__=="__main__":
    test_sparsemax()
