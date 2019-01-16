import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

from preprocess import Dataset
from lm import LM
from graph import Graph

from log_utils import get_logger
global logger


class ContextLMLoss(nn.Module):

    def __init__(self, context_sz):
        super(ContextLMLoss, self).__init__()
        self.D = context_sz
    
    
    def forward(self, Xhat, X):
        '''
        loss function for language model training.
            Xhat: [b, DxT,|V|]
            X: [b,T]
        loss = sum_t{CrossEntropy(Xhat[:,t*D:(t+1)*D,|V|], X[:,t+1:t+D+1]) for t=0...T-1
        '''
        T = X.shape[1]            # NOTE: X[:,-1] is <EOS>
        D = self.D
        losses = []
        for t in range(T-D):
            pred = Xhat[:,t*D:(t+1)*D,:]        # [b,D,|V|]
            l = F.cross_entropy(torch.transpose(pred,1,2), X[:,t:t+D])
            logger.debug('loss per context:%6.2f'%l)
            losses.append(l)
        return sum(losses)


class Trainer():
    def __init__(self, hparams, datadir,ckpt, _logger=None, gpu=None):
        self._hparams = hparams
        self._data = datadir
        self._logger = _logger
        self._gpu = gpu
        self._ckpt = ckpt



    def train(self, savedir):
        self._logger.info('Constructing model from data [%s] with hparams:\n%s...'\
                %(self._data, json.dumps(self._hparams['Model'],indent=4)))
        dataset = Dataset.load_ds(self._data)
        T = self._hparams['Model']['max_len'] # upper bound of input length -- different batch can have different T
        self.g = Graph(dataset.max_len(T), self._hparams['Model'])
        self._model = LM(dataset.vocab_sz, self._hparams['Model'], self.g)
        self._logger.info('Constructing optimizer...')
        criterion = ContextLMLoss(self._hparams['Model']['context_sz'])
        self._opt = torch.optim.Adam(self._model.parameters(), hparams['Trainer']['lr'])
        params = [(name, p.shape) for name, p in self._model.named_parameters()]
        self._logger.info('Optimizing parameters: %s'%str(params))
        start_epoch = 0
        best_loss = np.inf
        # if there's checkpoint
        if self._ckpt:
            logger.info('Loading checkpoint from %s...'%self._ckpt)
            checkpoint = torch.load(self._ckpt)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            self._model.load_state_dict(checkpoint['model'])
            self._opt.load_state_dict(checkpoint['optimizer'])

        if self._gpu:
            model.cuda(self._gpu)
            criterion.cuda(self._gpu)

        self._logger.info('Start training with best_loss %6.4f, \nhparams:\n %s'%(best_loss, json.dumps(hparams['Trainer'], indent=4)))
        for epoch in range(start_epoch, self._hparams['Trainer']['epoch']):
            self._logger.info('=> Train epoch %d'%epoch)
            data_iter = dataset.make_batch(self._hparams['Trainer']['batch_sz'],'train',T)
            losses = []
            for step, data in enumerate(data_iter):
                if self._gpu:
                    data = data.cuda(self._gpu)
                X= torch.autograd.Variable(torch.from_numpy(data), requires_grad=False)
                y_pred = self._model(X)         # X:[b,T], y_pred:[b,T, |V|]
                loss = criterion(y_pred, X)
                self._logger.debug('loss per batch = %f'%loss)
                losses.append(loss.item())

                nn.utils.clip_grad_norm(self._model.parameters(), 1)  # gradient clipping
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()

            loss_per_epoch = np.array(losses).mean()
            self._logger.info('epoch %d done, loss=%6.4f'%(epoch, loss_per_epoch))
            if loss_per_epoch < best_loss:
                best_loss = loss_per_epoch
                self.save(savedir, best_loss, epoch)
                self._logger.info('>>New best loss: %s. Model saved into %s/best/'%(best_loss, savedir))
            



    def save(self, savedir,loss,epoch):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self._logger.info('Saving model into %s...'%savedir)
        # save state of training, model, optimizer
        torch.save({'model':self._model.state_dict(),
            'optimizer':self._opt.state_dict(),
            'graph': self.g.state_dict(),
            'epoch': epoch, 'loss':loss}, savedir)
        # save the training/model hparams
        with open('%s/config.json'%savedir,'w') as f:
            json.dump(self._hparams, f, indent=4)






if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c','--config', default='./experiment/config.json',
                        help='path to config file, default ./experiment/config.json')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to last checkpoint model')
    parser.add_argument('-d', '--data_dir', default='data/wiki', 
                        help='path to data folder')
    parser.add_argument('-l', '--log_fname', default='', 
                        help='generate logs in fname.log')
    parser.add_argument('-s', '--save_dir', default='experiment/expr001', 
                        help='path to save trained model')
    parser.add_argument('-D', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    
    args = parser.parse_args()

    logger = get_logger(args.log_fname)
    hparams = json.load(open(args.config,'r'))
    if args.resume:
        logger.info('Overriding hparams from %s/config.json...'%self._ckpt)
        hparams = json.load(open('%s/config.json'%savedir,'r'))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    trainer = Trainer( hparams,args.data_dir,args.resume, logger, args.device)
    trainer.train(args.save_dir) 

