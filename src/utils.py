import subprocess
import json
import collections
import sys
import gc
from io import StringIO
import pandas as pd
import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt

def default_hparams():
    '''return a set of default hyperparams'''
    hparam_str = '''
        {
            "Trainer": {
                "epoch": 50,
                "lr": 1e-4,
                "batch_sz": 100,
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
    

