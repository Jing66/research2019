import numpy as np
import matplotlib.pyplot as plt
import math
import json
import argparse
import torch
import glob
import utils
import pdb


def plot_heap_map(ax, mma, target_labels, source_labels):
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(mma.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(mma.shape[0]) + 0.5, minor=False)
    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))
    ax.invert_yaxis()
    ax.set_xticklabels(source_labels, minor=False)
    ax.set_yticklabels(target_labels, minor=False)
    ax.tick_params(axis='x', rotation=90)
    plt.subplots_adjust(top=0.95, bottom=0.1)


def load_save_alignment(filedir):
    '''return alignmnet matrix (ie, attention) and snetence'''
    for fname in glob.glob('%s/*.pkl'%utils.format_dirname(filedir)):
        print("Saving weights from [%s]" %fname)
        savename=fname.split('.')[0]
        ckpt = torch.load(fname)
        labels, mm = ckpt['label'],ckpt['weights']
        fig, axies =  plt.subplots(math.ceil(mm.shape[0]/2),2,sharex='col',sharey='row',figsize=(20, 15))
        
        for l in range(mm.shape[0]):
            ax = axies[math.floor(l/2)][l%2]
            plot_heap_map(ax, mm[l,:,:],labels,labels)
        plt.savefig(savename+'.png')
        plt.close()






def __test():
    labels = ['Games','Minutes','Points','Field goals made','Field goal attempts','Field goal percentage','Free throws made','Free throws attempts','Free throws percentage','Three-pointers made','Three-point attempt','Three-point percentage','Offensive rebounds','Defensive rebounds','Total rebounds','Assists','Steals','Blocks','Turnover','Personal foul']
    mma = np.random.rand(len(labels),len(labels))
    fig, ax = plt.subplots()
    plot_heap_map(ax, mma,labels,labels)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Plot attention weights from saved files')
    parser.add_argument('attn_dirs', nargs='+',help='directories where attention weight are saved in')
    args = parser.parse_args()
    for d in args.attn_dirs:
        load_save_alignment(d)
