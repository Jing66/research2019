import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from datetime import timedelta
import pdb

from preprocess import Dataset
from lm import LM
from graph import Graph
import utils
from log_utils import get_logger
from lossFn import ContextLMLoss
global logger


class Trainer():
    def __init__(self, hparams, datadir,ckpt, _logger=None, gpu=None):
        self._hparams = hparams
        self._data = datadir
        self._logger = _logger
        self._gpu = gpu
        self._ckpt = ckpt


    def run_one_epoch(self, dataset, epoch):
        self._logger.info('=> Training epoch %d'%epoch)
        data_iter = dataset.make_batch(self._hparams['Trainer']['batch_sz'],'train',
                            self._hparams['Model']['max_len'] , # upper bound of input sentence length
                            self._hparams['Trainer']['total_samples'])  # total number of samples to train
        losses = 0
        for step, (data, data_lens) in enumerate(data_iter):
            # torch.cuda.empty_cache()
            data = torch.from_numpy(data).type(torch.LongTensor)
            data_lens = torch.from_numpy(data_lens).type(torch.LongTensor)
            X = torch.autograd.Variable(data, requires_grad=False)
            lens = torch.autograd.Variable(data_lens, requires_grad=False)
            if self._gpu is not None:
                X = X.cuda()
                lens = lens.cuda()
            self._opt.zero_grad()
            # y_pred = self._model(X, lens)         # X:[b,T], y_pred:[b,T, |V|]
            # loss = criterion(y_pred, X)
            loss = self._model(X, lens)
            self._logger.debug('loss per batch = %f'%loss)
            losses+=loss.detach().cpu().item()

            nn.utils.clip_grad_norm_(self._model.parameters(), 2)  # gradient clipping
            loss.backward()

            # debug: print gradients
            grad_of_param = {}
            for name, parameter in self._model.named_parameters():
                grad_of_param[name] = parameter.grad
                self._logger.debug('gradient of %s: \n%s'%(name, str(parameter.grad)))
            self._opt.step()
        loss_per_epoch = losses/step

        if  math.isnan(loss_per_epoch):
            self._logger.error("Get NaN loss for epoch %d-- exiting" %epoch)
            System.exit(1)
        return loss_per_epoch



    def train(self, savedir):
        self._logger.info("Loading data from [%s]..." %(self._data))
        dataset = Dataset.load_ds(self._data)
        self._logger.info(str(dataset))

        # build model, loss, optimizer
        self._logger.info("Constructing model with hparams:\n%s" %json.dumps(self._hparams['Model'],indent=4) )
        self.g = Graph( self._hparams['Model'], self._logger)
        vocab_cutoff = dataset.cutoff_vocab(self._hparams['Trainer']['vocab_clusters'])
        self._model = LM(dataset.vocab_sz, self._hparams['Model'], self.g, self._logger,vocab_cutoff )
        criterion = ContextLMLoss(self._hparams['Model']['Feature']['context_sz'], self._logger)
        self._logger.info('Constructing optimizer: %s' %self._hparams['Trainer']['optimizer'])
        optimizer = getattr(torch.optim, self._hparams['Trainer']['optimizer'])
        self._opt = optimizer(self._model.parameters(),self._hparams['Trainer']['lr'])
        params = [(name, p.shape) for name, p in self._model.named_parameters()]
        self._logger.debug('Optimizing parameters: %s'%str(params))
        start_epoch = 0
        best_loss = np.inf
        # if there's checkpoint
        if self._ckpt:
            logger.info('Loading checkpoint from %sexprt.ckpt...'%self._ckpt)
            if self._gpu is None:
                checkpoint = torch.load("%s/exprt.ckpt"%self._ckpt, map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load("%s/exprt.ckpt"%self._ckpt)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            self._model.load_state_dict(checkpoint['model'])
            self._opt.load_state_dict(checkpoint['optimizer'])
            self._logger.info("checkpoint experiment loaded")

        if self._gpu is not None:
            self.g.cuda()
            self._model.cuda()
            criterion.cuda()
            for state in self._opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        # training steps
        self._logger.info('Start training with best_loss %6.4f, \nhparams:\n %s'%(best_loss, json.dumps(hparams['Trainer'], indent=4)))
        for epoch in range(start_epoch, start_epoch + self._hparams['Trainer']['epoch']):
            epoch_start = time.time()
            loss_per_epoch = self.run_one_epoch(dataset, epoch)
            epoch_time = time.time() - epoch_start
            self._logger.info('epoch %d done, training time=[%s], training loss=[%6.4f]'%(epoch, str(timedelta(seconds=epoch_time)) , loss_per_epoch))


            # evaluate every epoch
            valid_start = time.time()
            loss_per_validate = self.validate(dataset)
            valid_time = time.time() - valid_start
            self._logger.info('eval on epoch %d done, eval time=[%s], dev loss=%6.4f'%(epoch,str(timedelta(seconds=valid_time)), loss_per_validate))

            # save best model on dev set
            if loss_per_validate < best_loss:
                best_loss = loss_per_validate
                self.save(savedir, best_loss, epoch)
                self._logger.info('>>New best validation loss: %s. Model saved into %s/exprt.ckpt'%(best_loss, savedir))
            


    def validate(self, dataset):
        self._logger.info("Start evaluating on dev set...")
        losses = 0
        data_iter_eval = dataset.make_batch(self._hparams['Trainer']['batch_sz'],'dev',
                            self._hparams['Model']['max_len'] ) # upper bound of input sentence length
        with torch.no_grad():
            for step, (data, data_lens) in enumerate(data_iter_eval):
                d = torch.from_numpy(data).type(torch.LongTensor)
                l = torch.from_numpy(data_lens).type(torch.LongTensor)
                X= torch.autograd.Variable(d, requires_grad=False)
                lens = torch.autograd.Variable(l, requires_grad=False)
                if self._gpu is  not None:
                    X = X.cuda()
                    lens = lens.cuda()
                # y_pred = self._model(X, lens)
                # loss = criterion(y_pred, X)
                loss = self._model(X, lens)
                losses += loss.detach().cpu().item()
        loss_per_epoch = losses/(step+1)
        return loss_per_epoch



    def save(self, savedir,loss,epoch):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self._logger.info('Saving model into %s/exprt.ckpt...'%savedir)
        # save state of training, model, optimizer
        torch.save({'model':self._model.state_dict(),
            'optimizer':self._opt.state_dict(),
            'graph': self.g.state_dict(),
            'epoch': epoch, 'loss':loss}, "%s/exprt.ckpt"%savedir)
        # save the training/model hparams
        with open('%s/config.json'%savedir,'w') as f:
            json.dump(self._hparams, f, indent=4)






if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c','--config', default='./experiment/config.json',
                        help='path to config file, default ./experiment/toy_config.json')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to last checkpoint model')
    parser.add_argument('-d', '--data_dir', default='data/out', 
                        help='path to data folder')
    parser.add_argument('-l', '--log_fname', default='train', 
                        help='generate logs in fname.log')
    parser.add_argument('-s', '--save_dir', default='experiment/test', 
                        help='path to save trained model')
    parser.add_argument('--device',type=int, default=None)
    parser.add_argument('--seed', default=999, type=int, help="torch random seed")
    parser.add_argument('--cpu', dest='cpu', default=False, action='store_true')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')

    args = parser.parse_args()

    logger = get_logger(args.log_fname, args.debug, args.save_dir)
    logger.info("Setting pytorch/numpy random seed to %s"%args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    free_gpu, free_mem  = utils.get_free_gpu()
    if free_mem > 1000 and not args.cpu and not args.device :
            args.device = free_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    logger.info("Using free GPU: %s"%args.device)

    torch.cuda.manual_seed(args.seed)
        
    if args.resume:
        args.resume += "/" if args.resume[-1]!="/" else ""
        logger.info('Overriding hparams from %s/config.json...'%args.resume)
        hparams = json.load(open('%s/config.json'%args.resume,'r'))
    else:
        default_hparams = utils.default_hparams()
        _hparams = json.load(open(args.config,'r'))
        hparams = utils.update_dict(default_hparams,_hparams)

    trainer = Trainer( hparams,args.data_dir,args.resume, logger,args.device)
    trainer.train(args.save_dir) 

