from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import json
import numpy as np
import os
import argparse
import itertools
from itertools import repeat
import matplotlib.pyplot as plt

from log_utils import get_logger
global logger

PAD = 0
EOS = 1
UNK = 2


def _pad(arr, size):
    return np.pad(arr, (0,size-arr.shape[0]), 'constant')


class Dataset():
    def __init__(self, sents=[], sent_bound=[0],vocab={'PAD':PAD, 'EOS':EOS,'UNK':UNK}):
        self._sents = sents
        self._sent_bound = sent_bound # indicates the boundary of sentences (within the same doc)
        self._vocab = vocab

    def build(self, doc):
        '''build or aggregate dataset from doc'''        
        sents = sent_tokenize(doc)
        if len(sents) == 0:
            return
        for sent in sents:
            words = word_tokenize(sent)
            sent_in_idx = []
            for w in words:
                if not w in self._vocab:
                    self._vocab[w] = len(self._vocab)+1
                widx = self._vocab[w]
                sent_in_idx.append(widx)
            self._sents.append(sent_in_idx)
        self._sent_bound.append(len(self._sents))



    def filter_docs(self, lower, upper):
        '''filter out docs with #sentence < lower or > upper, return new Dataset'''
        logger.info('--filtering docs with length (#sentence) out of range (%s, %s)' %(lower,upper))
        lens = self._sent_bound[1:] - self._sent_bound[:-1]
        valid = np.argwhere((lens>lower) & (lens<upper)).ravel() # idx for valid doc
        valid_bound = np.cumsum(lens[valid]) # sentence boundary for valid docs
        valid_bound = np.concatenate((np.zeros(1),valid_bound)) # always has 0 at front
        docs = [self.get_doc(i) for i in valid] # [doc1=[sent1,sent2...]]
        sents = list(itertools.chain.from_iterable(docs)) 
        # pdb.set_trace()
        return Dataset(sents,valid_bound,self._vocab)


    def filter_sents(self, lower, upper):
        logger.info('--filtering sentences with length (#words) out of range (%s, %s)' %(lower,upper))
        lens = np.array([len(s) for s in self._sents])
        valid = np.argwhere((lens>lower) & (lens<upper)).ravel() # idx for valid sentences
        invalid = np.argwhere((lens<=lower) | (lens>=upper)).ravel()
        sents = [self._sents[i] for i in valid]
        bins = np.digitize(invalid,self._sent_bound) - 1 
        bins[-1] = bins[-1]-1 if bins[-1]==len(self._sent_bound)-1 else bins[-1]
        slens = self._sent_bound[1:] - self._sent_bound[:-1]
        lsent = np.bincount(bins) 
        nlens = slens -  _pad(lsent,len(slens)) # how many sentences less per doc
        valid_bound = np.concatenate((np.zeros(1),np.cumsum(nlens)))
        # pdb.set_trace()
        return Dataset(sents, valid_bound, self._vocab)



    @classmethod
    def load_save_docs(cls, in_files, out_dir):
        '''return a Dataset class from a list of files.
            files are 1 article per line
        '''
        ds = Dataset()
        for fname in in_files:
            print("processing file: %s..."%fname)
            with open(fname,'r') as fp:
                lines = fp.readlines()
            for line in lines:
                ds.build(line)
        ds.save(out_dir)
        return ds


    def save(self, d):
        ''' save vocab(dict), sentences(list), sentence_boundary'''      
        if not os.path.exists(d):
            os.mkdir(d)
        serialized = pickle.dumps(self._sents)
        with open("%s/sents.pkl"%d,'wb') as file_object:
            file_object.write(serialized)
        with open('%s/vocab.json'%d, "w") as f:
            json.dump(self._vocab,f)
        np.save('%s/docs'%d, np.array(self._sent_bound))

    @classmethod
    def load_ds(cls, d):
        with open("%s/sents.pkl"%d,'rb') as f:
            serialized = f.read()
        sents = pickle.loads(serialized)
        with open('%s/vocab.json'%d, "r") as read_file:
            vocab = json.load(read_file)
        sent_bound = np.load('%s/docs.npy'%d)
        return Dataset(sents, sent_bound, vocab)

    def __str__(self):
        '''print out information about this dataset'''
        s = '===== INFO of Dataset =====\n'
        s += "dataset has %s documents,  %s sentences, %s vocabs"\
                    %(len(self._sent_bound)-1, self._sent_bound[-1], len(self._vocab))
        slen = np.array(self._sent_bound[1:]) - np.array(self._sent_bound[:-1])
        s += '\nDocs length (#sentences): avg %s, median %s, max %s, min %s'\
                        %(slen.mean(), np.median(slen), slen.max(), slen.min())
        wlen = np.array([len(s) for s in self._sents])
        s += '\nSentence length (#words): avg %s, median %s, max %s, min %s'\
                        %(wlen.mean(), np.median(wlen), wlen.max(), wlen.min())
        return s


    def __getitem__(self, index):
        return self._sents[index]
    
    def get_doc(self, idx):
        if idx > len(self._sent_bound-1):
            raise ValueError('doc #%d does not exist! '%idx)
        l,r = self._sent_bound[idx], self._sent_bound[idx+1]    
        return self._sents[int(l):int(r)]

    def plot_length(self,steps=50, fname='.'):
        slen = self._sent_bound[1:] - self._sent_bound[:-1]
        T = slen.max()
        x1 = np.linspace(0, T, num=steps, dtype=np.int64)
        digits = np.digitize(slen,x1)
        y1 = np.bincount(digits)

        fig, axes = plt.subplots(nrows=2, ncols=1)
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        plt.hist(y1, x1)
        plt.title('Distribution of document length')
        plt.ylabel('# doc')

        wlens = np.array([len(s) for s in self._sents])
        S = wlens.max()
        x2 = np.linspace(0, S, num=steps, dtype=np.int64)
        digits2 = np.digitize(wlens,x2)
        y2 = np.bincount(digits2)
        plt.subplot(2, 1, 2)
        plt.hist( y2,x2)
        plt.title('Distribution of sentence length')
        plt.ylabel('# sentence')

        plt.savefig('%s/dist.png'%fname)
        plt.close()

    def make_batch(self, bsize, shuffle=True, max_len=np.inf):
        '''make a batch of sentences, doesn't need to be in the same doc'''
        slen = self._sent_bound[1:] - self._sent_bound[:-1]
        max_len = min(max_len, np.max(slen)) # T
        sents = [s for s in self._sents if len(s)<max_len]
        if shuffle:
            idx = np.random.permutation(len(sents))
        start = 0
        while start + bsize < len(sents):
            batch = [np.array(sents[i]) for i in idx[start: start+bsize]]
            start += bsize
            padded = [_pad(b, max_len) for b in batch]
            out = np.array(padded).reshape((bsize,-1))
            yield out





def __test(args):
    logger.info("Testing with input_files" + str(args.input_files)+"out_dir:"+str(args.out_dir))
    in_files =args.input_files
    # ds = Dataset.load_save_docs(in_files, args.out_dir)
    logger.info("loading saved dataset")
    ds2 = Dataset.load_ds(args.out_dir)
    logger.info(ds2)
    #dg = ds2.make_batch(500, max_len=100)
    #while True:
    #    try:
    #        data = next(dg)
    #        print('data', data[:2])
    #    except StopIteration:
    #        print("data set all consumed!")
    #        break
    ds3 = ds2.filter_docs(100,500)
    print(ds3)
    #ds4 = ds2.filter_sents(5,60)
    #logger.info(ds4)
    #ds4.plot_length(fname=args.out_dir)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input_files', nargs='+',
                        help='list of files to process')
    parser.add_argument('-o','--out_dir', default='./',
                        help='directory to store the processed files. detault: ./')
    parser.add_argument('-t','--test', dest='test', action='store_true',
                        help='run the test function')
    parser.add_argument('-p','--process', dest='process', action='store_true',
                        help='save docs from input files')
    parser.add_argument('-l', '--log_fname', default='', 
                        help='generate logs in fname.log')
    args = parser.parse_args()

    logger = get_logger(args.log_fname)
    if args.test:
        __test(args)
    else:
        if not args.process:
            logger.info("Building datasets from "+str(args.input_files)+"; saving into" +str(args.out_dir))
            ds = Dataset.load_save_docs(args.input_files, args.out_dir)
        else:
            logger.info('Loading and processing datasets from %s'%args.out_dir)
            ds = Dataset.load_ds(args.out_dir)
            logger.info(ds)
            ds_filtered = ds.filter_docs(3,500)
            # pdb.set_trace()
            # ds_filtered = ds.filter_sents(3,150)
            logger.info(ds_filtered)
            ds_filtered.save(args.out_dir+'_filtered')
            ds_filtered.plot_length(fname=args.out_dir+'_filtered')
