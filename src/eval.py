import argparse
import torch
import json
import time
from datetime import timedelta
import os
import numpy as np

import utils
from log_utils import get_logger
from train import Trainer


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model ')
    parser.add_argument('model', help="Model class to evaluate")
    parser.add_argument('ckpt_dir', help='path to saved checkpoint model')
    parser.add_argument('data_dir', help='path to test data folder')
    parser.add_argument('-l', '--log_fname', default='eval', 
                        help='generate logs in fname.log')
    parser.add_argument('--device',type=int, default=None)
    parser.add_argument('--seed', default=999, type=int, help="torch random seed")
    parser.add_argument('--cpu', dest='cpu', default=False, action='store_true')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')

    args = parser.parse_args()

    logger = get_logger(args.log_fname, args.debug, args.ckpt_dir)
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

    default_hparams = utils.default_hparams(args.model)
    _hparams = json.load(open('%s/config.json'%utils.format_dirname(args.ckpt_dir),'r'))
    hparams = utils.update_dict(default_hparams,_hparams)
    hparams['Trainer']['model_output'] = 'logprobs'
    hparams['Model']['Feature']['SS_prob'] = 1.0

    if args.model.lower()=="glomo":
        trainer = Trainer(hparams,utils.format_dirname(args.data_dir), utils.format_dirname(args.ckpt_dir), logger,args.device, test_only=True)
    elif args.model.lower()=='baselinelm':
        trainer = BaseLMTrainer( hparams, utils.format_dirname(args.data_dir),  utils.format_dirname(args.resume),logger,args.device)
    else:
        raise ValueError('Model type %s unsupported'%args.model)

    trainer.load_ckpt()
    start = time.time() 
    loss, accuracy = trainer.validate('test')
    test_time = time.time() - start
    logger.info('=> Evaluation done! test time=[%s], loss=%6.4f, accuracy=%6.4f'%(str(timedelta(seconds=test_time)),loss, accuracy))
