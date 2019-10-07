from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from collections import Counter, OrderedDict


def log_softmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return np.log(e_x / e_x.sum(axis=dim, keepdims=True))


class RenormalizeVocabPost(object):
    def __init__(self, vocab_map_fn=None, vocab_fn=None, special=[],
                 unique_flag='head'):

        self.special = special
        self.vocab_map_fn = vocab_map_fn
        self.load_vocab(vocab_fn)
        # the sentence piece style is 'head' and the bpe style is 'end'
        self.unique_flag = unique_flag

    def build_renormalize_vocab(self):
        self.load_vocab_mapping(self.vocab_map_fn)
        self.build_trie_tree()

    def set_cuda(self, cuda):
        pass

    def build_trie_tree(self):
        self.trie = {}
        head_vocab_set = {}
        for i in range(self.head_vocab.shape[0]):
            if self.head_vocab[i] == 1:
                head_vocab_set[i] = {}
        for word in self.vocab_map.keys():
            node = self.trie
            for i in range(len(self.vocab_map[word])):
                token = self.vocab_map[word][i]
                if token not in node:
                    node[token] = {}
                assert not (self.unique_flag=='head' and i>0 \
                    and self.head_vocab[token]==1), \
                    'middle token cannot be head token'
                assert not (self.unique_flag=='end' and
                    i < len(self.vocab_map[word])-1 \
                    and self.end_vocab[token]==1), \
                    'middel token cannot be end token'
                node = node[token]
            for token in self.special_idx:
                node[token] = {}
        for token in self.special_idx:
            self.trie[token] = head_vocab_set

    def load_vocab_mapping(self, map_fn):
        self.head_vocab = np.zeros(len(self.idx2sym), dtype=int)
        self.end_vocab = np.zeros(len(self.idx2sym), dtype=int)
        self.vocab_map = {}
        with open(map_fn) as fin:
            for i, line in enumerate(fin):
                tokens = line.strip().split('\t')
                sub_tokens_idx = [self.sym2idx[token]
                    for token in tokens[1].split()]
                self.vocab_map[tokens[0]] = sub_tokens_idx
                self.head_vocab[sub_tokens_idx[0]] = 1
                self.end_vocab[sub_tokens_idx[-1]] = 1

        for token in self.special_idx:
            self.head_vocab[token] = 1
            self.end_vocab[token] = 1

        assert self.check_reconstructable(map_fn)

    def check_reconstructable(self, map_fn):
        D = {}
        with open(map_fn) as fin:
            for line in fin:
                tokens = line.strip().split()
                sub_tokens = ' '.join(tokens[1:])
                if sub_tokens in D:
                    print('warning: similar sub-token representation: ',
                        tokens[0], D[sub_tokens], tokens[1:])
                else:
                    D[sub_tokens] = tokens[0]
        return True

    def load_vocab(self, vocab_file):
        print('load vocab from {}'.format(vocab_file))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        for token in self.special:
            self.add_symbol(token)
        self.special_idx = [self.sym2idx[token] for token in self.special]

    def build_initial_status(self, bsz):
        status = [-1] * bsz
        return status

    def get_eval_loss(self, data, target, out_logit, last_status):
        '''
            input tensor: numpy array (T X bsz)
            target tensor: numpy array (T X bsz)
            output prob: numpy array (T X bsz X V)
        '''
        T, bsz, V = out_logit.shape
        next_word, new_status = self.get_next_word_set(data, last_status)
        head_vocab = self.head_vocab.reshape((1,1,-1))
        end_marks = self.get_end_marks(data).reshape(T, bsz, 1)
        head_vocab = head_vocab * end_marks
        next_word = next_word | head_vocab
        out_logit[next_word == 0] = -float('inf')
        logit = log_softmax(out_logit, dim=2)
        target = target.reshape(T, bsz, 1)
        nll = -np.take_along_axis(logit, target, axis=2)
        return nll, new_status

    def get_end_marks(self, symbols):
        ret = np.take(self.end_vocab, symbols)
        return ret

    def get_next_word_set(self, symbols, last_status):
        next_word = np.zeros(
            (symbols.shape[0], symbols.shape[1], len(self.sym2idx)), dtype=int)
        new_status = []
        for batch_idx in range(symbols.shape[1]):
            node = last_status[batch_idx]
            for i in range(symbols.shape[0]):
                token = symbols[i][batch_idx]
                if self.unique_flag == 'head':
                    if self.head_vocab[token] == 1:
                        node = self.trie
                if node == -1:
                    next_word[i][batch_idx][:] = 1
                else:
                    token_list = list(node[token].keys())
                    next_word[i][batch_idx][token_list] = 1
                    if self.unique_flag == 'end' and self.end_vocab[token] == 1:
                        node = self.trie
                    else:
                        node = node[token]
            new_status.append(node)
        return next_word, new_status

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            assert sym in self.sym2idx, 'The unk token is prohibited'

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices
                             if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
