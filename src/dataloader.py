import torch.utils.data as data
from preprocess import Dataset
import numpy as np
import torch

import pdb

def get_data_loader(dataset, mode, batch_size=100, max_len=np.inf, max_sample=np.inf, n_workers=0):
    '''dataset: Dataset class
        mode: `train`/`dev`/`test`'''
    if mode=='test':
        indices = np.arange(len(dataset._test))+len(dataset._train)+len(dataset._dev)
    elif mode=='dev':
        indices = np.arange(len(dataset._dev))+len(dataset._train)
    else:
        indices = np.arange(len(dataset._train))
    if max_sample < len(indices):
        indices = indices[:max_sample]    
    sampler = data.SubsetRandomSampler(indices)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=n_workers, 
                            collate_fn=lambda b:collate_fn(b,max_len))
    return data_loader




def collate_fn(batch, max_len):
    '''
   Args:
        - batch: list of sentences, each of tensor shape (?)
        - max_len: int, output should have max length this number
    ''' 
    def _merge(seqs, max_len):
        batch_lens = torch.LongTensor([len(d) for d in seqs])
        if max_len > batch_lens[0]:
            max_len = batch_lens[0]
        else:
            batch_lens[batch_lens > max_len] = max_len
        padded_seqs = torch.zeros(len(seqs), max_len).long()
        for i, seq in enumerate(seqs):
            end = batch_lens[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, batch_lens

    batch.sort(key=lambda x: len(x), reverse=True)
    seqs, lens = _merge(batch, max_len)
    return seqs, lens
    





def test(d):
    ds = Dataset.load_ds(d)
    for _ in range(5):
        print("Testing train data loader...")
        tl= get_data_loader(ds,"train", 50)
        for batch_index, (data, lens) in enumerate(tl):
            pass
        vl = get_data_loader(ds, "dev")
        print("Testing dev data loader...")
        for  batch_index, (data, lens) in enumerate(vl):
            pass
        print("Testing test data loader...")
        tl = get_data_loader(ds, "test")
        for batch_index, (data, lens) in enumerate(tl):
            pass

if __name__=='__main__':
    test('data/toy')
