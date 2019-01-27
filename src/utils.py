import subprocess
import json
import collections
import sys
import gc
from io import StringIO
import pandas as pd
import numpy as np
import torch
import re
import pdb
import matplotlib.pyplot as plt
import argparse

def default_hparams():
    '''return a set of default hyperparams'''
    hparam_str = '''
        {
            "Trainer": {
                "epoch": 50,
                "lr": 0.005,
                "train_batch_sz": 200,
                "eval_batch_sz": 50,
                "optimizer": "SGD",
                "total_samples": 8000000,
                "save_period": 5,
                "model_output": "loss",
                "vocab_clusters": 5
            },
            "Model": {
                "max_len": 100,
                "n_layers": 6,
                "embd_sz": 300,
                "dropout": 0.5,
                "Graph": {
                    "sparsity_fn": "leaky_relu",
                    "kernel_sz": 7,
                    "linear_feat": 100,
                    "n_filter_k": 20,
                    "n_filter_q": 20
                },
                "Feature": {
                    "context_sz": 5,
                    "compose_fn": "GRUCell",
                    "SS_prob":0.0
                }
            }
        }
        '''
    return json.loads(hparam_str)



def get_free_gpu():
    '''return the idx and free memory of the most free GPU'''
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf-8')),
                     names=['memory.used', 'memory.free'],skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = pd.to_numeric(gpu_df['memory.free']).idxmax()
    print('Returning GPU:{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx, int(gpu_df.iloc[idx]['memory.free'])


def get_mask_2d(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask

def get_mask_3d(seq_batch, mask_idx=0):
    batch_sz = seq_batch.size()[0]
    max_len = seq_batch.size()[1]
    mask = torch.ones(batch_sz, max_len, max_len)
    mask[seq_batch==mask_idx] = 0.0
    mask2 = torch.transpose(mask,1,2)
    mask2[seq_batch==mask_idx] = 0.0
    return torch.autograd.Variable(mask2)



def memReport():
    count=0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
            count+=1
    print("total", count)

def format_dirname(dirname):
    '''return dirname as a string ending without "/"'''
    if dirname is None:
        return dirname
    if dirname[-1]!='/':
        return dirname
    else:
        return dirname[:-1]

def update_dict(default, dict2):
    '''update default dict with another dict'''
    for k, v in dict2.items():
        if isinstance(v, collections.Mapping):
            default[k] = update_dict(default.get(k, {}), v)
        else:
            default[k] = v
    return default


def plot_train_dev_metrics(train_losses, dev_losses, metric_name,  fname=None):
    train_losses, dev_losses = np.array(train_losses), np.array(dev_losses)
    plt.plot(train_losses, 'r-', label='training '+metric_name)
    plt.plot(dev_losses, 'b-', label='dev '+metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.title(metric_name)
    plt.legend(loc='upper right')
    if fname:
        plt.savefig(fname+'.png')
    else:
        plt.show()
    plt.close()
    


def parse_logs(fnames):
    '''return train/dev losses/accuracy from a list of files'''
    train_losses, dev_losses = {}, {}
    train_accuracy, dev_accuracy = {},{}
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                epoch = re.search(r'epoch \d+ done',line)
                if epoch:
                    nb_epoch = int(epoch.group().split()[1])
                    if re.search(r'training', line):
                        isTrain = True
                    elif re.search(r'validation',line):
                        isTrain = False
                    loss_str=re.findall(r'loss=[0-9].[0-9]+',line)
                    if len(loss_str)>0:
                        loss = float(loss_str[0].split('=')[-1])
                    acc_str = re.findall(r'accuracy =\s+[0-9].[0-9]+',line)
                    if len(acc_str)>0:
                        accuracy = float(acc_str[0].split('=')[-1].strip())
                    if isTrain:
                        train_losses[nb_epoch] = loss
                        train_accuracy[nb_epoch] = accuracy
                    else:
                        dev_losses[nb_epoch] = loss
                        dev_accuracy[nb_epoch] = accuracy
    def _sort_by_key(dicts):
        return [dicts[k] for k in sorted(dicts.keys())]
    return _sort_by_key(train_losses), _sort_by_key(train_accuracy), _sort_by_key(dev_losses), _sort_by_key(dev_accuracy)

if __name__=='__main__':
    # parse log files and plot metrics
    parser = argparse.ArgumentParser(description='Plot metrics from log files')
    parser.add_argument('log_files', nargs='+',help='path to log files')
    parser.add_argument('-s', '--save_dir', default='./', 
                        help='directory of the saved plot')
    parser.add_argument('--loss', default=False,action='store_true', help='plot loss graph')
    parser.add_argument('--acc', default=False, action='store_true', help='plot accuracy graph')
    args = parser.parse_args()

    t_l, t_a, d_l, d_a = parse_logs(args.log_files)
    if args.loss:
        plot_train_dev_metrics(t_l, d_l, 'loss', format_dirname(args.save_dir)+'/loss')
    if args.acc:
        plot_train_dev_metrics(t_a, d_a, 'Accuracy', format_dirname(args.save_dir)+'/accuracy')
