import torch
from torchnlp.word_to_vector import GloVe
from torchnlp.datasets import imdb_dataset
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
import pickle
import json

from util.log_utils import  get_logger
global logger
import pdb
from preprocess import Dataset, normalize_str

PAD = 0
EOS = 1
START = 2
UNK = 3

UNK_THRES = 5
TRAIN_DEV_RATIO = [8,2]
LABELS = {'pos':1,'neg':0}

class IMDBData(Dataset):
    def __init__(self,train=[], dev=[], test=[],
                    vocab={'PAD':PAD, 'EOS':EOS,'START':START,'UNK':UNK}, w_freq=None):
        self.embd=None
        self.n_class = 2
        super(IMDBData,self).__init__(train,dev,test,vocab,w_freq)


    @classmethod
    def load_save_docs(cls, out_dir):
        train = imdb_dataset(train=True)
        test = imdb_dataset(test=True)
        train_ = []
        test_ = []
        for td in train:
            sent = normalize_str(td['text'])
            tup = (sent, LABELS[td['sentiment']])
            train_.append(tup)
        for td in test:
            sent = normalize_str(td['text'])
            tup = (sent, LABELS[td['sentiment']])
            test_.append(tup)
        ds = IMDBData()
        ds.build(train_, test_)
        ds.save(out_dir)


    def build(self, trains, tests):
        ratio = np.array(TRAIN_DEV_RATIO).astype(np.float32)
        ratio/= np.sum(ratio)
        train_len = int(ratio[0]*len(trains))
        word2freq = {}    
        vocab = self._vocab.copy() 
        for idx in range(len(trains)):
            sent, label = trains[idx]
            words = word_tokenize(sent)
            for w in words:
                word2freq[w] = word2freq.get(w,0)+1
        word2freq = sorted(word2freq.items(), key=lambda kv:-kv[1])
        w_freq = []
        for w, freq in word2freq:
            if freq > UNK_THRES:
                self._vocab[w] = len(self._vocab)  
                w_freq.append(freq)   
        self.w_freq = np.array(w_freq)

        def _append_to_self(tuples, to_append):
            sent, label = tuples
            sent_idx = [START] 
            for w in word_tokenize(sent):
                sent_idx.append(self._vocab.get(w,UNK))
            sent_idx.append(EOS)
            to_append.append((sent_idx, label))

        _ = [_append_to_self(t, self._train) for t in trains[:train_len]]
        _ = [_append_to_self(t, self._dev) for t in trains[train_len:]]
        _ = [_append_to_self(t, self._test) for t in tests]


        

    def __getitem__(self,index):
        sents = self._train + self._dev + self._test  
        seq = torch.tensor(sents[index][0]).long()
        label = torch.tensor(sents[index][1])
        return seq,label



    @classmethod
    def load_ds(cls, d, test_only=False):
        '''test_only: bool. If true, only load test dataset'''
        train, dev = [],[]
        if not test_only:
            with open("%s/train.pkl"%d,'rb') as f:
                serialized = f.read()
            train = pickle.loads(serialized)
            with open("%s/dev.pkl"%d,'rb') as f:
                serialized = f.read()
            dev = pickle.loads(serialized)
        with open("%s/test.pkl"%d,'rb') as f:
            serialized = f.read()
        test = pickle.loads(serialized)
        with open('%s/vocab.json'%d, "r") as read_file:
            vocab = json.load(read_file)
        w_freq = np.load('%s/w_freq.npy'%d)

        ds = IMDBData(train,dev,test, vocab, w_freq)
        ds.load_embedding(d)
        return ds

    
    def __str__(self):
        s = '===== INFO of IMDB Dataset =====\n'
        s += 'Dataset size (#sentences): train--%s, dev--%s, test--%s.'\
                    %(len(self._train),len(self._dev),len(self._test))
        s += '\nVocab size obtained from training set: %s'%self.vocab_sz
        alllens = np.array([len(s) for (s,_) in self._train+self._test+self._dev])
        s += '\nDataset sentence length info: max--%s, min--%s, mean--%s, median--%s'\
                %(alllens.max(), alllens.min(), np.mean(alllens), np.median(alllens))
        if self.embd is not None:
            s += 'Dataset contains embeddings'
        return s
        

def make_embedding(vocab, d):
    '''craete and save a (|V|xd) embedding matrix'''
    vectors = GloVe()
    dim = vectors['hi'].shape[0]
    embeddings = []
    for word,_ in vocab.items():
        if word in {'UNK','START','EOS'}:
            vec = np.random.randn(dim)
        vec = vectors[word]
        embeddings.append(vec)
    embd = torch.stack(embeddings)
    torch.save(embd, '%s/embd.pt'%d)


def __test(args):
    logger.info("Testing with out_dir:"+str(args.out_dir))
    ds = IMDBData.load_save_docs( args.out_dir)
    logger.info("loading saved dataset")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('out_dir',
                        help='directory to store the processed files. detault: ./')
    parser.add_argument('-t','--test', dest='test', action='store_true',
                        help='run the test function')
    parser.add_argument('-p','--process', dest='process', action='store_true')
    parser.add_argument('-embd','--embedding', dest='embedding', action='store_true',
                        help='make glove embeddings')
    parser.add_argument('-l', '--log_fname', default='', 
                        help='generate logs in fname.log')
    args = parser.parse_args()

    logger = get_logger(args.log_fname)
    if args.test:
        __test(args)
    else:
        if not args.process:
            logger.info("Building datasets, saving into " +str(args.out_dir))
            ds = IMDBData.load_save_docs(args.out_dir)
            logger.info(ds)
        else:
            logger.info('Loading and processing datasets from %s'%args.out_dir)
            ds = IMDBData.load_ds(args.out_dir)
            logger.info(ds)
        
        if args.embedding:
            make_embedding(ds._vocab, args.out_dir)
