import subprocess
import sys
import gc
from io import StringIO
import pandas as pd
import numpy as np
import torch
import pdb

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
