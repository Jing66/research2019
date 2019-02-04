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
from dataloader import get_data_loader 
from lm import LM
from baseline_lm import BaseLM
from graph import Graph
import utils
from log_utils import get_logger
from lossFn import ContextLMLoss, accuracy_fn
global logger


PAD=0


def calc_acc(logprobs, y):
    '''
    - logprobs: [b,T*(D-1),|V|]
    '''
    _, pred = torch.max(logprobs,2)
    tot_valid, tot_correct = accuracy_fn(pred,y)
    return float(tot_correct)/tot_valid



class Trainer(object):
    def __init__(self, config, datadir,ckpt, _logger=None, gpu=None, test_only=False):
        self.config = config
        self.logger = _logger
        self._gpu = gpu
        self._ckpt = ckpt
        self.build(datadir, test_only)


    def _build_models(self):
        vocab_cutoff = self.dataset.cutoff_vocab(self.config['Trainer']['vocab_clusters'])
        self.g = Graph( self.config['Model'], self.logger)
        self._model = LM(self.dataset.vocab_sz, self.config['Model'], self.g, self.logger,vocab_cutoff)
        self.criterion = ContextLMLoss(self.config['Model']['Feature']['context_sz'], self.logger)
        self.accuracy_fn = self.criterion.accuracy
        self._models = {'graph':self.g, 'model':self._model}


    def build(self, datadir, test_only=False):
        self.logger.info("Building trainer class %s" %self.__class__.__name__)
        self.logger.info("Loading data from [%s]..." %(datadir))
        self.dataset = Dataset.load_ds(datadir, test_only)
        self.logger.info(str(self.dataset))

        # build model, loss, optimizer
        self.logger.info("Constructing model with hparams:\n%s" %(json.dumps(self.config['Model'],indent=4) ))

        self._build_models()

        self.logger.info('Constructing optimizer: %s' %self.config['Trainer']['optimizer'])
        optimizer = getattr(torch.optim, self.config['Trainer']['optimizer'])
        self._opt = optimizer(self._model.parameters(),self.config['Trainer']['lr'])
        params = [(name, p.shape) for name, p in self._model.named_parameters()]
        self.logger.debug('Optimizing parameters: %s'%str(params))


    def load_ckpt(self):
        '''if there's checkpoint, return previous epoch and best loss'''
        best_loss = np.inf
        start_epoch = 0
        if self._ckpt:
            self.logger.info('Loading checkpoint from %s/exprt.ckpt...'%self._ckpt)
            if self._gpu is None:
                checkpoint = torch.load("%s/exprt.ckpt"%self._ckpt, map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load("%s/exprt.ckpt"%self._ckpt)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            self._model.load_state_dict(checkpoint['model'])
            self._opt.load_state_dict(checkpoint['optimizer'])
            self.logger.info("checkpoint experiment loaded, model trained until epoch %d, best_loss=%6.4f" %(start_epoch,best_loss))

        if self._gpu is not None:
            for m in self._models.values():
                m.cuda()
            self.criterion.cuda()
            for state in self._opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        return start_epoch, best_loss


    def attn_visualize(self, idx, savedir):
        '''save a file of attention weights for visualization'''
        data = self.dataset[idx]
        labels = self.dataset.idx2str(data)
        if self._gpu is not None:
            data = data.cuda()
        with torch.no_grad():
            attn_wgt = self._model.attn_fn(data).cpu()         # [b,L,T,T]
        attn_wgt = torch.squeeze(attn_wgt,0)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save({'label':labels, 'weights':attn_wgt},'%s/attn_%d.pkl'%(savedir,idx))
        self.logger.info("Attention weights for %dth sentence saved in [%s]"%(idx,savedir))
    
    
    def forward_pass(self, X, lens, output_probs,kwargs):
        ''' run one forward pass of the model, return accuracy and loss'''
        X = X[:,1:]             # ignore SOS
        lens = lens - 1
        y_pred = self._model(X, lens, output_probs)         # X:[b,T], y_pred:[b,T*D,|V|]
        if y_pred.dim():
            loss = self.criterion(torch.transpose(y_pred,1,2), X)
            acc = self.accuracy_fn(y_pred, X)
            return loss, acc, None
        else:
            loss = y_pred
            return loss, 0.0, None
            
        
    def forward_args(self):
        return None

    def run_one_epoch(self,  epoch, output_probs):
        '''
        Train one epoch of the whole dataset.
        Args:
            - dataset: class Dataset.
            - otuput_probs: if True, model output log probs, in which case we also calculate accuracy
        '''
        self._model.train()
        self.logger.info('=> Training epoch %d'%epoch)
        data_iter = get_data_loader(self.dataset,"train", self.config['Trainer']['train_batch_sz'],
                        max_len=self.config['Model']['max_len'] , # upper bound of input sentence length
                        max_sample=self.config['Trainer']['total_samples'],  # total number of samples to train
                        n_workers=self.config['Trainer']['n_workers'])
        kwargs = self.forward_args()
        losses, accuracies = 0,0
        for step, (data, data_lens) in enumerate(data_iter):
            X = torch.autograd.Variable(data, requires_grad=False)
            lens = torch.autograd.Variable(data_lens, requires_grad=False)
            if self._gpu is not None:
                X = X.cuda()
                lens = lens.cuda()
            self._opt.zero_grad()

            loss, acc, kwargs = self.forward_pass(X, lens, output_probs,kwargs)

            self.logger.debug('loss per batch = %f'%loss)
            losses+=loss.detach().cpu().item()
            accuracies += acc

            nn.utils.clip_grad_norm_(self._model.parameters(), 2)  # gradient clipping
            loss.backward()

            # debug: print gradients
            grad_of_param = {}
            for name, parameter in self._model.named_parameters():
                grad_of_param[name] = parameter.grad
                # self.logger.debug('gradient of %s: \n%s'%(name, str(parameter.grad)))
            self._opt.step()
        loss_per_epoch = losses/(step+1)
        acc_per_epoch = accuracies/(step+1)
    
        if  math.isnan(loss_per_epoch):
            self.logger.error("Get NaN loss for epoch %d-- exiting" %epoch)
            System.exit(1)

        return loss_per_epoch, acc_per_epoch



    def train(self, savedir):
        # training steps
        start_epoch, best_loss = self.load_ckpt()
        save_period = self.config['Trainer']['save_period']
        output_probs = True if self.config['Trainer']['model_output'] == 'logprobs' else False
        train_losses, dev_losses, train_accs, dev_accs  = [], [], [], []
        self.logger.info('Start training with best_loss %6.4f, \nhparams:\n %s'%(best_loss, json.dumps(hparams['Trainer'], indent=4)))
        for epoch in range(start_epoch, start_epoch + self.config['Trainer']['epoch']):
            # train one epoch, forward and backward on whole dataset
            epoch_start = time.time()
            loss_per_epoch, accuracies_per_epoch  = self.run_one_epoch( epoch, output_probs)
            epoch_time = time.time() - epoch_start
            self.logger.info('epoch %d done, training time=[%s], training loss=%6.4f, training accuracy = %6.3f'%(epoch, str(timedelta(seconds=epoch_time)) , loss_per_epoch, accuracies_per_epoch))
            train_losses.append(loss_per_epoch)
            train_accs.append(accuracies_per_epoch)

            # save periodically
            if epoch % save_period ==0:
                self.save("%s/ckpt_%d"%(savedir, epoch), best_loss, epoch)

            # evaluate every epoch
            valid_start = time.time()
            loss_per_validate, accuracy_per_validate  = self.validate()
            valid_time = time.time() - valid_start
            self.logger.info('validation on epoch %d done, eval time=[%s], dev loss=%6.4f, dev accuracy = %6.4f'%(epoch,str(timedelta(seconds=valid_time)), loss_per_validate, accuracy_per_validate))
            dev_losses.append(loss_per_validate)
            dev_accs.append(accuracy_per_validate)

            # save best model on dev set
            if loss_per_validate < best_loss:
                best_loss = loss_per_validate
                self.save('%s/best'%savedir, best_loss, epoch)
                self.logger.info('>>New best validation loss: %s. Model saved into %s/best/'%(best_loss, savedir))
                
            
        # Done -- plot graphs
        utils.plot_train_dev_metrics(train_losses, dev_losses,"loss", savedir+'/losses')
        if output_probs:
            utils.plot_train_dev_metrics(train_accs, dev_accs, "accuracy", savedir+'/accuracies')
        self.logger.info('==> Training Done!! best validation loss: %6.4f. Model/log/plots saved in [%s]' %(best_loss, savedir))


    def validate(self,  ds_name='dev'):
        self.logger.info("Setting model to eval mode. Start evaluating on %s set..." %ds_name)
        self._model.eval()
        losses, accuracies = 0, 0
        data_iter_eval = get_data_loader(self.dataset, ds_name, self.config['Trainer']['eval_batch_sz'],
                            max_len=self.config['Model']['max_len'], # upper bound of input sentence length
                            n_workers=self.config['Trainer']['n_workers'])
        kwargs = self.forward_args()
        with torch.no_grad():
            for step, (data, data_lens) in enumerate(data_iter_eval):
                X= torch.autograd.Variable(data, requires_grad=False)
                lens = torch.autograd.Variable(data_lens, requires_grad=False)
                if self._gpu is not None:
                    X = X.cuda()
                    lens = lens.cuda()

                loss, acc,kwargs = self.forward_pass(X, lens, True, kwargs)
                accuracies += acc
                losses += loss.detach().cpu().item()
        loss_per_epoch = losses/(step+1)
        return loss_per_epoch, accuracies/(step+1)



    def save(self, savedir,loss,epoch):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.logger.info('Saving model into %s/exprt.ckpt...'%savedir)
        save_dict = {'optimizer':self._opt.state_dict(),'epoch': epoch, 'loss':loss}
        for m_name, model in self._models.items():
            save_dict[m_name]=model.state_dict()
        # save state of training, model, optimizer
        torch.save(save_dict, "%s/exprt.ckpt"%savedir)
        # save the training/model hparams
        with open('%s/config.json'%savedir,'w') as f:
            json.dump(self.config, f, indent=4)


class BaseLMTrainer(Trainer):
    def __init__(self, config, datadir,ckpt, _logger=None, gpu=None, test_only=False):
        super(BaseLMTrainer,self).__init__(config, datadir,ckpt, _logger, gpu, test_only)

    def forward_args(self):
        return {"hidden":None}
    
    def forward_pass(self, X, lens,output_probs, kwargs):
        X = X[:,1:]
        lens = lens-1
        last_hidden = kwargs['hidden']
        acc = 0.0
        if output_probs:
            logprobs, hidden  = self._model(X, lens, last_hidden, output_probs)         # X:[b,T], y_pred:[b,T*D,|V|]
            loss = self.criterion(torch.transpose(logprobs,1,2), X)
            acc = self.accuracy_fn(logprobs, X)
        else:
            loss, hidden = self._model(X, lens, last_hidden, output_probs)
        hn = BaseLM.repackage(hidden)
        return loss, acc, {'hidden':hn}
    
    def _build_models(self):
        vocab_cutoff = self.dataset.cutoff_vocab(self.config['Trainer']['vocab_clusters'])
        self._model = BaseLM(self.dataset.vocab_sz, self.config['Model'], self.logger, vocab_cutoff)
        self.criterion = nn.NLLLoss(ignore_index=PAD)
        self.accuracy_fn = calc_acc
        self._models = {'model':self._model}





if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('model', default="GLoMo", help="which model to train")
    parser.add_argument('-c','--config', default=None,
                        help='path to config file, default ./experiment/toy_config.json')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to last checkpoint model')
    parser.add_argument('-d', '--data_dir', default='data/toy', 
                        help='path to data folder')
    parser.add_argument('-l', '--log_fname', default='train', 
                        help='generate logs in fname.log')
    parser.add_argument('-s', '--save_dir', default='experiment/test', 
                        help='path to save trained model')
    parser.add_argument('-v','--visualize',nargs='+', default=None, 
                        help='indices for selecting visualizations')
    parser.add_argument('--no-train', dest='no_train', default=False, action='store_true')
    parser.add_argument('--device',type=int, default=None)
    parser.add_argument('--seed', default=999, type=int, help="torch random seed")
    parser.add_argument('--cpu', dest='cpu', default=False, action='store_true')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')

    args = parser.parse_args()
    # setup logger, seed, gpu etc
    logger = get_logger(args.log_fname, args.debug, args.save_dir)
    logger.info("Setting pytorch/numpy random seed to %s"%args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    free_gpu, free_mem  = utils.get_free_gpu()
    if free_mem > 1000 and not args.cpu and not args.device :
            args.device = free_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    logger.info("Using free GPU: %s"%args.device)
    if args.device is not None:
        torch.cuda.manual_seed(args.seed)
        
    # overwrite config files from checkpoint
    _hparams={}
    if args.resume:
        logger.info('Overriding hparams from %s/config.json...'%utils.format_dirname(args.resume))
        _hparams = json.load(open('%s/config.json'%utils.format_dirname(args.resume),'r'))
    elif args.config:
        try:
            _hparams = json.load(open(args.config,'r'))
        except IOError:
            logger.error("Cannot load file %s, using default"%args.config)
    default_hparams = utils.default_hparams(args.model.lower())
    hparams = utils.update_dict(default_hparams,_hparams)

    if args.model.lower()=="glomo":
        trainer = Trainer( hparams, utils.format_dirname(args.data_dir),  utils.format_dirname(args.resume),logger,args.device)
    elif args.model.lower()=='baselinelm':
        trainer = BaseLMTrainer( hparams, utils.format_dirname(args.data_dir),  utils.format_dirname(args.resume),logger,args.device)
    else:
        raise ValueError('Model type %s unsupported'%args.model)

    if not args.no_train:
        trainer.train(args.save_dir) 
    if args.visualize:
        for idx in args.visualize:
            _,_ = trainer.load_ckpt()
            trainer.attn_visualize(int(idx), args.save_dir)

