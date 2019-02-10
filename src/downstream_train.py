import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import argparse
import time
from datetime import timedelta
import pdb

from train import Trainer
from imdb_preprocess import IMDBData
from dataloader import get_data_loader 
from models.graph import Graph
from models.classif import Classifier
from util import utils
from util.log_utils import get_logger
global logger


PAD=0

class ClassifierTrainer(Trainer):
    def __init__(self, config, datadir, ckpt,graph_ckpt, _logger=None, gpu=None, test_only=False):
        self.graph_ckpt = graph_ckpt
        super(ClassifierTrainer,self).__init__(config, datadir,ckpt, _logger, gpu, test_only)
    
    def calc_acc(self, logit, ytrue):
        _, idx = torch.max(logit, dim=-1)
        n_correct = torch.sum(idx==ytrue).item()
        n_total = ytrue.shape[0]
        return float(n_correct)/n_total
        
    def l2_penalty(self, x):
        xT = torch.transpose(x,1,2)
        identity = torch.eye(x.size(1),device=x.device)
        identity = Variable(identity.unsqueeze(0).expand(x.shape[0],x.size(1),x.size(1)))
        penal = self._model.l2_matrix_norm(x@xT - identity)
        return penal

    def build(self, datadir, test_only=False):
        self.logger.info("Building trainer class %s" %self.__class__.__name__)
        self.logger.info("Loading data from [%s]..." %(datadir))
        self.dataset = IMDBData.load_ds(datadir, test_only)
        self.logger.info(str(self.dataset))

        # build model, loss, optimizer
        self._build_models()

        self.logger.info('Constructing optimizer: %s' %self.config['Trainer']['optimizer'])
        optimizer = getattr(torch.optim, self.config['Trainer']['optimizer'])
        self._opt = optimizer(self._model.parameters(),self.config['Trainer']['lr'])
        params = [(name, p.shape) for name, p in self._model.named_parameters()]
        self.logger.debug('Optimizing parameters: %s'%str(params))

    def _build_models(self):
        # load frozen graph
        if not self.graph_ckpt and self.config['Model'].get('Graph') is None:
            raise Exception("Must provide a trained graph for downstreaming task")
        if self.graph_ckpt is not None:  
            graph_hparams = json.load(open('%s/config.json'%self.graph_ckpt,'r'))
            self.config['Model']= utils.update_dict(self.config['Model'],graph_hparams['Model'])
        self.logger.info("Complete hparams with graph:\n%s"%(json.dumps(self.config['Model'],indent=4)))
        self._graph = Graph(self.config['Model'], self.logger)
        if self.graph_ckpt is not None:
            logger.info("Loading graph from checkpoint %s"%self.graph_ckpt)
            if self._gpu is None:
                checkpoint = torch.load("%s/exprt.ckpt"%self.graph_ckpt, map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load("%s/exprt.ckpt"%self.graph_ckpt)
            try:
                self._graph.load_state_dict(checkpoint['graph'])
            except Exception as e:
                self.logger.error(e)
                self.logger.error("Failed to load graph")
        # build classifier model
        self._model = Classifier(self.dataset.vocab_sz, self.config['Model'],
                self.dataset.n_class, self._graph, self.logger, self.dataset.embd)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_fn = self.calc_acc
        self._models = {'model':self._model}


    def forward_args(self):
        return {'hidden': None}

    def forward_pass(self, data, lens, kwargs):
        X, ytrue = data
        X= torch.autograd.Variable(X, requires_grad=False)
        lens = torch.autograd.Variable(lens, requires_grad=False)
        if self._gpu is not None:
            X = X.cuda()
            lens = lens.cuda()
            ytrue = ytrue.cuda()
        logits, hidden, attention  = self._model(X, lens, kwargs['hidden'])
        penal = self.l2_penalty(attention)
        C = self.config["Trainer"]["penalty"]
        loss = self.criterion(logits, ytrue) + C* penal/X.shape[0]
        acc = self.accuracy_fn(logits,ytrue)
        hn = Classifier.repackage(hidden)
        return loss, acc, {"hidden":hn}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('model', default="GLoMo", help="which model to train")
    parser.add_argument('data_dir', help='path to data folder')
    parser.add_argument('-c','--config', default=None,
                        help='path to config file, default ./experiment/toy_config.json')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to last checkpoint model')
    parser.add_argument('-l', '--log_fname', default='train', 
                        help='generate logs in fname.log')
    parser.add_argument('-s', '--save_dir', default='experiment/test', 
                        help='path to save trained model')
    parser.add_argument('-v','--visualize',nargs='+', default=None, 
                        help='indices for selecting visualizations')
    parser.add_argument('-g','--graph_dir', default=None, help='ckpt dir to load graph')
    parser.add_argument('--no-train', dest='no_train', default=False, action='store_true')
    parser.add_argument('--device',type=int, default=None)
    parser.add_argument('--seed', default=999, type=int, help="torch random seed")
    parser.add_argument('--cpu', dest='cpu', default=False, action='store_true')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')

    args = parser.parse_args()
    # setup logger, seed, gpu etc
    logger = get_logger(args.log_fname, args.debug, args.save_dir)
    logger.info("Command line args: $"+(" ").join(sys.argv))
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
            logger.error("Cannot load file %s, using default hparams!!"%args.config)
    default_hparams = utils.default_hparams(args.model.lower())
    hparams = utils.update_dict(default_hparams,_hparams)

    if args.model.lower()=='imdb':
        trainer = ClassifierTrainer(hparams,  utils.format_dirname(args.data_dir), 
                    utils.format_dirname(args.resume),  
                    utils.format_dirname(args.graph_dir), logger, args.device) 
    else:
        raise ValueError('Model type %s unsupported'%args.model)

    if not args.no_train:
        trainer.train(args.save_dir) 
    if args.visualize:
        _,_ = trainer.load_ckpt()
        for idx in args.visualize:
            trainer.attn_visualize(int(idx), args.save_dir)


