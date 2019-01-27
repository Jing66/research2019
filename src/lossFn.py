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
            pred = logprobs[:,t*D:(t+1)*D,:]        # [b,D,|V|]
            l = F.nll_loss(torch.transpose(pred,1,2), X[:,t:t+D], ignore_index = PAD)
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
            valids = (X[:,t:t+D]!=PAD)
            Xhat_t = Xhat[:, t*D:(t+1)*D]
            num_correct = torch.sum(X[:, t:t+D][valids] == Xhat_t[valids])
            tot_correct += num_correct.detach().cpu().item()
            tot_valids += torch.sum(valids).detach().cpu().item()
        return tot_correct/(tot_valids)
