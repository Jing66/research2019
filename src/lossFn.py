import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

PAD = 0

class ContextLMLoss(nn.Module):

    def __init__(self, context_sz, logger=None):
        super(ContextLMLoss, self).__init__()
        self.D = context_sz
        self.logger = logger
    
    
    def forward(self, logprobs, X):
        '''
        loss function for language model training.
            logprobs: [b, DxT,|V|]
            X: [b,T]
        loss = sum_t{CrossEntropy(Xhat[:,t*D:(t+1)*D,|V|], X[:,t+1:t+D+1]) for t=0...T-1
        '''
        T = X.shape[1]            # NOTE: X[:,-1] is <EOS>
        D = self.D
        losses = []
        for t in range(T-D+1):
            pred = logprobs[:,:,t*D:(t+1)*D]        # [b,|V|, D]
            l = F.nll_loss(pred, X[:,t:t+D], ignore_index = PAD)
            self.logger.debug('loss per context L(Xhat[%d:%d], X[%d:%d]):%6.2f'%(t*D, (t+1)*D, t,t+D,l))
            losses.append(l)
        return sum(losses)/len(losses)

    @torch.no_grad() 
    def accuracy(self, logprobs, X):
        _, Xhat = torch.max(logprobs, dim=2)
        T = X.shape[1]            # NOTE: X[:,-1] is <EOS>
        D = self.D
        tot_correct = 0
        tot_valids = 0
        for t in range(T-D+1):
            Xhat_t = Xhat[:, t*D:(t+1)*D]
            num_valids, num_correct = accuracy_fn(Xhat_t, X[:,t:t+D])
            tot_correct += num_correct
            tot_valids += num_valids
        return tot_correct/(tot_valids)


def accuracy_fn(Xhat, X):
    valids = X.ne(PAD)
    num_correct = torch.sum(torch.masked_select(Xhat,valids) == torch.masked_select(X,valids))
    return torch.sum(valids).item(), num_correct.item()
