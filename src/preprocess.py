from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import re
import json
import numpy as np
import os
import argparse
import itertools
import matplotlib.pyplot as plt

from log_utils import  get_logger
global logger

PAD = 0
EOS = 1
UNK = 2
UNK_THRES = 5       # if a word appears <UNK_THRES times, make it UNK
RATIO = [7.5,1.5,1]     # ratio of train/dev/test split
SENT_THRES = 5      # if a sentence has < SNET_THRES words, ignore it


def _pad_or_trunc(arr, size):
    if arr.shape[0] < size:
        return np.pad(arr, (0,size-arr.shape[0]), 'constant')
    else:
        return arr[:size]

def normalize_str(s):
    '''normalize string:
        - convert to lower case
        - replace digits with 5 (so `1` and `4` will be treated the same, but treated different from `12`
    '''
    s = s.lower()
    s = re.sub(r'(\n)+','\n',s)
    s = re.sub(r'[0-9]', '5',s)
    return s


class Dataset():
    def __init__(self, train=[], dev=[], test=[] ,vocab={'PAD':PAD, 'EOS':EOS,'UNK':UNK}):
        self._train = train
        self._test = test
        self._dev = dev
        self._vocab = vocab
        self._vocab_sz = len(vocab)

    @property
    def vocab_sz(self):
        return len(self._vocab)

    @vocab_sz.getter
    def vocab_sz(self):
        return len(self._vocab)

    def max_len(self, upper):
        ''' return the length of longest sentence in dataset that is lower than upper'''
        slens = np.array([len(s) for s in self._train])
        return min(upper, slens.max())

    def __len__(self):
        return len(self._train)

    #def filter_docs(self, lower, upper):
    #    '''filter out docs with #sentence < lower or > upper, return new Dataset'''
    #    logger.info('--filtering docs with length (#sentence) out of range (%s, %s)' %(lower,upper))
    #    lens = self._sent_bound[1:] - self._sent_bound[:-1]
    #    valid = np.argwhere((lens>lower) & (lens<upper)).ravel() # idx for valid doc
    #    valid_bound = np.cumsum(lens[valid]) # sentence boundary for valid docs
    #    valid_bound = np.concatenate((np.zeros(1),valid_bound)) # always has 0 at front
    #    docs = [self.get_doc(i) for i in valid] # [doc1=[sent1,sent2...]]
    #    sents = list(itertools.chain.from_iterable(docs)) 
    #    # pdb.set_trace()
    #    return Dataset(sents,valid_bound,self._vocab)


    #def filter_sents(self, lower, upper):
    #    logger.info('--filtering sentences with length (#words) out of range (%s, %s)' %(lower,upper))
    #    lens = np.array([len(s) for s in self._sents])
    #    valid = np.argwhere((lens>lower) & (lens<upper)).ravel() # idx for valid sentences
    #    invalid = np.argwhere((lens<=lower) | (lens>=upper)).ravel()
    #    sents = [self._sents[i] for i in valid]
    #    bins = np.digitize(invalid,self._sent_bound) - 1 
    #    bins[-1] = bins[-1]-1 if bins[-1]==len(self._sent_bound)-1 else bins[-1]
    #    slens = self._sent_bound[1:] - self._sent_bound[:-1]
    #    lsent = np.bincount(bins) 
    #    nlens = slens -  _pad(lsent,len(slens)) # how many sentences less per doc
    #    valid_bound = np.concatenate((np.zeros(1),np.cumsum(nlens)))
    #    # pdb.set_trace()
    #    return Dataset(sents, valid_bound, self._vocab)



    @classmethod
    def load_save_docs(cls, in_files, out_dir):
        '''return a Dataset class from a list of files.
            files are 1 article per line
        '''
        ds = Dataset()
        sents = []
        for fname in in_files:
            logger.info("adding sentences from: `%s`..."%fname)
            with open(fname,'r') as fp:
                lines = fp.readlines()
            for doc in lines:
                if len(doc)>0:
                    normed = normalize_str(doc)
                    sents.extend(sent_tokenize(normed))
        ds.build(sents)
        ds.save(out_dir)
        return ds

    
    def build(self,sents):
        ''' sents: [sent1, sents2...]. each sentence has type str.'''
        ratio = np.array(RATIO).astype(np.float32)
        ratio/= np.sum(ratio)
        train_len = int(ratio[0]*len(sents))
        dev_len = int(ratio[1]*len(sents))
        # shuffle all sentences randomly before splitting into train/dev/test
        shuffled_idx = np.random.permutation(len(sents))
        word2freq = {}
        vocab = self._vocab.copy()
        for idx in range( train_len + dev_len):
            words = word_tokenize(sents[shuffled_idx[idx]])
            if len(words) < SENT_THRES:
                dev_len -=1
                continue                # sentence too short
            for w in words:
                word2freq[w] = word2freq.get(w,0)+1
        # filter out infrequent words in vocab, remap vocab
        for w, freq in word2freq.items():
            if freq > UNK_THRES:
                self._vocab[w] = len(self._vocab)

        # train -- convert to idx and map to vocab
        logger.info('Processing training set with new vocab')
        for idx in range(len(sents)):
            if len(self._train) > train_len:
                break
            words = word_tokenize(sents[shuffled_idx[idx]])
            if len(words) < SENT_THRES:
                continue                # sentence too short
            sent_idx = []
            for w in words:
                sent_idx.append(self._vocab.get(w,UNK))
            sent_idx.append(EOS)
            self._train.append(sent_idx)


        logger.info('Processing dev/test set...')
        # dev/test -- convert to idx
        for j in range(idx, len(sents)):
            words = word_tokenize(sents[shuffled_idx[j]])
            if len(words) < SENT_THRES:
                continue
            sent_idx = []
            for w in words:
                sent_idx.append(self._vocab.get(w,UNK))
            sent_idx.append(EOS)
            if len(self._dev) < dev_len:
                self._dev.append(sent_idx)
            else:
                self._test.append(sent_idx)



    def save(self, d):
        ''' save vocab(dict), train/dev/test'''      
        if not os.path.exists(d):
            os.mkdir(d)
        serialized = pickle.dumps(self._train)
        with open("%s/train.pkl"%d,'wb') as file_object:
            file_object.write(serialized)
        serialized = pickle.dumps(self._dev)
        with open("%s/dev.pkl"%d,'wb') as file_object:
            file_object.write(serialized)
        serialized = pickle.dumps(self._test)
        with open("%s/test.pkl"%d,'wb') as file_object:
            file_object.write(serialized)
        with open('%s/vocab.json'%d, "w") as f:
            json.dump(self._vocab,f)

    @classmethod
    def load_ds(cls, d):
        with open("%s/train.pkl"%d,'rb') as f:
            serialized = f.read()
        train = pickle.loads(serialized)
        with open("%s/test.pkl"%d,'rb') as f:
            serialized = f.read()
        test = pickle.loads(serialized)
        with open("%s/dev.pkl"%d,'rb') as f:
            serialized = f.read()
        dev = pickle.loads(serialized)
        with open('%s/vocab.json'%d, "r") as read_file:
            vocab = json.load(read_file)
        return Dataset(train,dev,test, vocab)

    def __str__(self):
        '''print out information about this dataset'''
        s = '===== INFO of Dataset =====\n'
        s += 'Dataset size (#sentences): train--%s, dev--%s, test--%s.'\
                    %(len(self._train),len(self._dev),len(self._test))
        s += '\nVocab size obtained from training set: %s'%self.vocab_sz
        alllens = np.array([len(s) for s in self._train+self._test+self._dev])
        s += '\nDataset sentence length info: max--%s, min--%s, mean--%s, median--%s'\
                %(alllens.max(), alllens.min(), np.mean(alllens), np.median(alllens))
        tlens = alllens[:len(self._train)]
        s += '\nTraining sentence length info: max--%s, min--%s, mean--%s, median--%s'\
                %(tlens.max(), tlens.min(), np.mean(tlens), np.median(tlens))
        s += '\n Training corpus #token :%d' %(np.sum(tlens))
        return s


    def __getitem__(self, index):
        return self._sents[index]
    

    def plot_length(self,steps=50, fname='.'):
        all_sents = self._train + self._test + self._dev
        wlens = np.array([len(s) for s in all_sents])
        S = wlens.max()
        x2 = np.linspace(0, S, num=steps, dtype=np.int64)
        digits2 = np.digitize(wlens,x2)
        y2 = np.bincount(digits2)
        plt.hist( y2,x2)
        plt.title('Distribution of sentence length')
        plt.ylabel('# sentences')
        plt.xlabel('# words (bins)')

        plt.savefig('%s/dist.png'%fname)
        plt.close()




    def make_batch(self, bsize, mode, shuffle=True, max_len=np.inf):
        '''make a batch of padded sentences with length up to max_len. If too long, truncate
            -- mode: str, "train"/"dev"/"test"
        '''
        sents = getattr(self,"_"+mode)
        # slens = [len(s) for s in sents]
        if shuffle:
            idx = np.random.permutation(len(sents))
        start = 0
        while start < len(sents):
            batch = [np.array(sents[i]) for i in idx[start: start+bsize]]
            start += bsize
            # FIXME:T can differ for each training batch?
            T = np.array([len(b) for b in batch]).max()
            max_len = min(max_len, T)
            padded = [_pad_or_trunc(b, max_len) for b in batch]
            out = np.array(padded).reshape((-1,max_len))
            yield out





def __test(args):
    logger.info("Testing with input_files" + str(args.input_files)+"out_dir:"+str(args.out_dir))
    in_files =args.input_files
    ds = Dataset.load_save_docs(in_files, args.out_dir)
    logger.info("loading saved dataset")
    ds2 = Dataset.load_ds(args.out_dir)
    logger.info(ds2)
    dg = ds2.make_batch(200, 'test',max_len=100)
    while True:
        try:
            data = next(dg)
            print('data', data.shape, data[0], data[-1])
        except StopIteration:
            print("data set all consumed!")
            break
    ds2.plot_length(fname=args.out_dir)



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
            logger.info("Building datasets from "+str(args.input_files)+"; saving into " +str(args.out_dir))
            ds = Dataset.load_save_docs(args.input_files, args.out_dir)
            print(ds)
        else:
            logger.info('Loading and processing datasets from %s'%args.out_dir)
            ds = Dataset.load_ds(args.out_dir)
            logger.info(ds)
            ds.plot_length(fname=args.out_dir)
