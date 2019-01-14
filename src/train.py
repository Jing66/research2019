import json
import torch
import torch.nn as nn
import os

from preprocess import Dataset
from lm import LM


from log_utils import get_logger
global logger


class Trainer():
    def __init__(self, hparams, datadir,ckpt, _logger=None, gpu=None):
        self._hparams = hparams
        self._data = datadir
        self._logger = _logger
        self._gpu = gpu
        self._ckpt = ckpt



    def train(self):
        self._logger.info('Constructing model from data [%s] with hparams:\n%s...' %(self._data, str(self._hparams['Model'])))
        dataset = Dataset.load_ds(self._data)
        self._model = LM(self._hparams['Model'])
        self._logger.info('Constructing optimizer...')
        criterion = nn.CrossEntropyLoss()
        self._opt = torch.optim.Adam(self._model.parameters(), hparams['Trainer']['lr'])
        if self._ckpt:
            logger.info('Loading checkpoint from %s...'%self._ckpt)
            checkpoint = torch.load(self._ckpt)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            self._model.load_state_dict(checkpoint['model'])
            self._opt.load_state_dict(checkpoint['optimizer'])

        if self._gpu:
            model.cuda()
            criterion.cuda()
        # TODO: train
        self._logger.info('Start training with hparams:\n %s'%(str(hparams['Trainer'])))



        self._logger.info('Training done. best loss: %s. Saving model into %s/best/'%(best_loss, savedir))
        self.save(savedir, best_loss, epoch)





    def save(self, savedir,loss,epoch):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self._logger.info('Saving model into %s...'%savedir))
        # save state of training, model, optimizer
        torch.save({'model':self._model.state_dict(),
            'optimizer':self._opt.state_dict(),
            'epoch', epoch+1, 'loss':loss}, savedir)






if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c','--config', default='./experiment/config.json',
                        help='path to config file, default ./experiment/config.json')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='path to last checkpoint model')
    parser.add_argument('-d', '--data_dir', default='data/wiki_all', 
                        help='path to data folder')
    parser.add_argument('-l', '--log_fname', default='', 
                        help='generate logs in fname.log')
    parser.add_argument('-s', '--save_dir', default='experiment/expr001', 
                        help='path to save trained model')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    
    args = parser.parse_args()

    logger = get_logger(args.log_fname)
    hparams = json.load(open(args.config,'r'))
    model = LM(hparams)
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    trainer = Trainer( hparams, args.data_dir,args.resume, logger, args.device)
    trainer.train()

