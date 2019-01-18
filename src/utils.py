import subprocess
import sys
from io import StringIO
import pandas as pd
import numpy as np

def get_free_gpu():
    '''return the idx and free memory of the most free GPU'''
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf-8')),
                     names=['memory.used', 'memory.free'],skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    idx = pd.to_numeric(gpu_df['memory.free']).idxmax()
    print('Returning GPU:{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx, int(gpu_df.iloc[idx]['memory.free'])



