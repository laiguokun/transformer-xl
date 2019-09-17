import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab

class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len      

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMOrderedRenomalizeIterator(object):
    def __init__(self, data, bsz, bptt, vocab, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.device = device
        self.vocab = vocab
    
        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data[0].size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        self.data = data[0].narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = self.data.view(bsz, -1).t().contiguous().to(device)
        
        self.next_word = data[1]
        self.end_mark = data[2].narrow(0, 0, self.n_step * bsz)
        self.end_mark = self.end_mark.view(bsz, -1).t().contiguous()

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

        self.head_vocab = vocab.head_vocab.view(1, 1, -1).expand(1, bsz, -1)

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i+1:i+1+seq_len]
        next_word_batch = torch.zeros(target.size(0), target.size(1), len(self.vocab)).long()
        for row in range(i, i+seq_len):
            for col in range(self.bsz):
                original_pos = col * self.data.size(0) + row 
                next_word_batch[row - i][col][self.next_word[original_pos]] = 1
        head_vocab = self.head_vocab.expand(target.size(0), self.bsz, -1)
        end_mark = self.end_mark[i:i+seq_len].unsqueeze(-1)
        head_vocab = head_vocab * end_mark
        next_word_batch = next_word_batch | head_vocab
        '''
        print(data[:][9])
        for row in range(i, i+seq_len):
            for col in range(self.bsz):
                if (next_word_batch[row - i][col][target[row-i][col]] == 0):
                    print(row, col, i)
                    assert False
        '''
        next_word_batch = next_word_batch.byte().to(self.device)
        return data, [target, next_word_batch], seq_len      


    def get_fixlen_iter(self, start=0):
        
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
        shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, path, dataset, only_eval=False, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)
        self.only_eval = only_eval

        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103':
            if self.only_eval ==False:
                self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called
        self.vocab.build_vocab()
        self.renormalize = kwargs['renormalize']
        if self.renormalize:
            self.vocab.build_renormalize_vocab(path)

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            if self.only_eval == False:
                self.train = self.vocab.encode_file(
                    os.path.join(path, 'train.txt'), ordered=True)
            if self.renormalize:
                self.valid = self.vocab.encode_file_renormalize(
                    os.path.join(path, 'valid.txt'), os.path.join(path, 'valid_next_word.pt'),
                    ordered=True)
                self.test  = self.vocab.encode_file_renormalize(
                    os.path.join(path, 'test.txt'), os.path.join(path, 'test_next_word.pt'),
                    ordered=True)
            else:
                self.valid = self.vocab.encode_file(
                    os.path.join(path, 'valid.txt'), ordered=True)
                self.test  = self.vocab.encode_file(
                    os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                if self.renormalize:
                    kwargs['vocab'] = self.vocab
                    data_iter = LMOrderedRenomalizeIterator(data, *args, **kwargs)
                else:
                    data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                data_iter = LMShuffledIterator(data, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset, renormalize=False, only_eval=False):
    if renormalize:
        fn = os.path.join(datadir, 'renormalize_cache.pt')
    else:
        fn = os.path.join(datadir, 'cache.pt')
    if os.path.exists(fn) and False:
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset {}...'.format(dataset))
        kwargs = {}
        if dataset in ['wt103', 'wt2']:
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            pass
        kwargs['renormalize'] = renormalize
        if only_eval:
            kwargs['vocab_file'] = os.path.join(datadir, 'train-vocab.txt')
        corpus = Corpus(datadir, dataset, only_eval=only_eval, **kwargs)
        #torch.save(corpus, fn)

    return corpus

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/wikitext-103-sp',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()
    corpus = get_lm_corpus(args.datadir, args.dataset, renormalize=True, only_eval=True)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
    va_iter = corpus.get_iterator('valid', 10, 64, device='cpu', ext_len=0)
    for idx, (data, target, seq_len) in enumerate(va_iter):
        print(target[0].size())
        print(target[1].size())
        print(target[1][0][0].sum())
        break
