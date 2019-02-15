import torch
import torch.nn as nn

# taken from https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, inputs):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = inputs.size()
        inputs = inputs.view(-1, inputs.size(self.dim))
        
        dim = 1
        number_of_logits = inputs.size(dim)

        # Translate input by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        ranges = torch.arange(start=1, end=number_of_logits+1,dtype=zs.dtype, device=inputs.device).view(1, -1)
        ranges = ranges.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + ranges * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * ranges, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(inputs), inputs - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

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
