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



    def train(self, savedir):
        self._logger.info('Constructing model from data [%s] with hparams:\n%s...' %(self._data, str(self._hparams['Model'])))
        dataset = Dataset.load_ds(self._data)
        T = self._hparams['Model']['max_len'] # upper bound of input length -- different batch can have different T
        self.g = Graph(dataset.max_len(T), self._hparams['Model'])
        self._model = LM(dataset.vocab_sz, self._hparams['Model'], self.g)

        self._logger.info('Constructing optimizer...')
        criterion = nn.CrossEntropyLoss()
        self._opt = torch.optim.Adam(self._model.parameters(), hparams['Trainer']['lr'])
        params = [(name, p.shape) for name, p in self._model.named_parameters]
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

        self._logger.info('Start training with best_loss %d, \nhparams:\n %s'%(best_loss, str(hparams['Trainer'])))
        for epoch in range(start_epoch, self._hparams['Trainer']['epoch']):
            self._logger.info('=> Train epoch %d'%epoch)
            data_iter = dataset.make_batch(self._hparams['Trainer']['batch_sz'],'train',T)
            losses = []
            for step, data in enumerate(data_iter):
                if self._gpu:
                    data = data.cuda(self._gpu)

                y_pred = self._model(data) # data:[b,T], y_pred:[b,T, |V|]
                # TODO: modify loss function
                loss = criterion(y_pred, data)
                print('loss per batch',loss)
                losses.append(loss.item())

                nn.utils.clip_grad_norm(self._model.parameters(), 1)  # gradient clipping
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()

            loss_per_epoch = np.array(losses).mean()
            self._logger.info('epoch %d done, loss='%(epoch, loss_per_epoch))
            if loss_per_epoch < best_loss:
                best_loss = loss_per_epoch
                self.save(savedir, best_loss, epoch)
                self._logger.info('>>New best loss: %s. Model saved into %s/best/'%(best_loss, savedir))
            



    def save(self, savedir,loss,epoch):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self._logger.info('Saving model into %s...'%savedir))
        # save state of training, model, optimizer
        torch.save({'model':self._model.state_dict(),
            'optimizer':self._opt.state_dict(),
            'graph': self.g.state_dict(),
            'epoch', epoch, 'loss':loss}, savedir)
        # save the training/model hparams
        with open('%s/config.json'%savedir,'w') as f:
            json.dump(self._hparams, f, indent=4)






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
    if self._ckpt:
        logger.info('Overriding hparams from %s/config.json...'%self._ckpt)
        hparams = json.load(open('%s/config.json'%savedir,'r'))

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    trainer = Trainer( hparams,args.data_dir,args.resume, logger, args.device)
    trainer.train(args.save_dir) 

