from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import json
import numpy as np
import os
import argparse



PAD = 0
EOS = 1
UNK = 2

class Dataset():
    def __init__(self, sents=[], sent_bound=[0],vocab={'PAD':PAD, 'EOS':EOS,'UNK':UNK}):
        self._sents = sents
        self._sent_bound = sent_bound # indicates the boundary of sentences (within the same doc)
        self._vocab = vocab

    def __str__(self):
        return "dataset has %s sentences, %s vocabs"\
                    %(len(self._sent_bound)-1, len(self._vocab))

    def build(self, doc):
        '''build or aggregate dataset from doc'''        
        sents = sent_tokenize(doc)
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

    @classmethod
    def load_save_docs(cls, in_files, out_dir):
        '''return a Dataset class from a list of files.
            files are 1 article per line
        '''
        ds = Dataset()
        for fname in in_files:
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



def __test(args):
    print("Testing with input_files", args.input_files, "out_dir",args.out_dir)
    in_files =args.input_files
    ds = Dataset.load_save_docs(in_files, args.out_dir)
    print(ds)
    print("loading saved dataset")
    ds2 = Dataset.load_ds(args.out_dir)
    print(ds2)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_files', nargs='+',
                        help='list of files to process')
    parser.add_argument('-o','--out_dir', default='./',
                        help='directory to store the processed files. detault: ./')

    args = parser.parse_args()
    # __test(args)
    print("Building datasets from ", args.input_files, "saving into ", args.out_dir)
    ds = Dataset.load_save_docs(args.input_files, args.out_dir)
    print(ds)
