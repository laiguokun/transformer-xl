import os
from collections import Counter, OrderedDict

import torch

class RenormalizeVocab(object):
    def __init__(self, vocab_map_fn=None, vocab_fn=None, add_eos=True,
            add_double_eos=False, delimiter=None, lower_case=True, special=[]):

        self.lock = False
        self.delimiter = delimiter
        self.lower_case = lower_case
        self.add_double_eos = add_double_eos
        self.add_eos = add_eos
        self.special = special
        self.vocab_map_fn = vocab_map_fn
        self.load_vocab(vocab_fn)

    def build_renormalize_vocab(self, ):
        self.lock = True
        self.load_vocab_mapping(self.vocab_map_fn)
        self.build_trie_tree()

    def build_trie_tree(self,):
        self.trie = {}
        head_vocab_set = {}
        for i in range(self.head_vocab.size(0)):
            if self.head_vocab[i] == 1:
                head_vocab_set[i] = {}
        for word in self.vocab_map.keys():
            node = self.trie
            for i in range(len(self.vocab_map[word])):
                token = self.vocab_map[word][i]
                if token not in node:
                    node[token] = {}
                assert not (i>0 and self.head_vocab[token]==1), 'middle token cannot be head token'
                node = node[token]
            for token in self.special_idx:
                node[token] = {}
        for token in self.special_idx:
            self.trie[token] = head_vocab_set

    def tokenize(self, line):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if self.add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif self.add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def load_vocab_mapping(self, map_fn):
        self.head_vocab = torch.zeros(len(self.idx2sym)).byte()
        self.end_vocab = torch.zeros(len(self.idx2sym)).byte()
        self.vocab_map = {}
        with open(map_fn) as fin:
            for i, line in enumerate(fin):
                if self.lower_case:
                    line = line.lower()
                tokens = line.strip().split('\t')
                sub_tokens_idx = [self.sym2idx[token] for token in tokens[1].split()]
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
                    print('warning: similar sub-token representation: ', tokens[0], D[sub_tokens], tokens[1:])
                else:
                    D[sub_tokens] = tokens[0]
        return True

    def load_vocab(self, vocab_file):
        print('load vocab from {}'.format(vocab_file))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.lower_case:
                    line = line.lower()
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.special_idx = [self.sym2idx[token] for token in self.special]

    def encode_file(self, path, ordered=False, verbose=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded_symbols = []
        encoded_candidates = []
        encoded_end_marks = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line)
                symbols = self.convert_to_tensor(symbols)
                candidates = self.get_next_word_set(symbols)
                assert symbols.size(0) == len(candidates)
                end_mark = self.get_end_marks(symbols)
                encoded_symbols.append(symbols)
                encoded_candidates.append(candidates)
                encoded_end_marks.append(end_mark)
        encoded = [encoded_symbols, encoded_candidates, encoded_end_marks]
        if ordered:
            encoded[0] = torch.cat(encoded[0])
            encoded[1] = [item for sublist in encoded_candidates for item in sublist]
            encoded[2] = torch.cat(encoded[2])
        return encoded

    def get_end_marks(self, symbols):
        ret = torch.zeros_like(symbols).byte()
        for idx in range(symbols.size(0)):
            if self.end_vocab[symbols[idx]] == 1:
                ret[idx] = 1
        return ret

    def get_next_word_set(self, symbols):
        next_word_set = []
        node = self.trie
        symbols = symbols.tolist()
        for i in range(len(symbols)):
            if self.head_vocab[symbols[i]] == 1:
                node = self.trie
            node = node[symbols[i]]
            next_word_set.append(torch.LongTensor(list(node.keys())))
        return next_word_set

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

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

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)

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
        next_word_batch = torch.zeros(target.size(0), target.size(1), len(self.vocab)).byte()
        for row in range(i, i+seq_len):
            for col in range(self.bsz):
                original_pos = col * self.data.size(0) + row 
                next_word_batch[row - i][col][self.next_word[original_pos]] = 1
        head_vocab = self.head_vocab.expand(target.size(0), self.bsz, -1)
        end_mark = self.end_mark[i:i+seq_len].unsqueeze(-1)
        head_vocab = head_vocab * end_mark
        next_word_batch = next_word_batch | head_vocab
        next_word_batch = next_word_batch.to(self.device)
        return data, [target, next_word_batch], seq_len      


    def get_fixlen_iter(self, start=0):
        
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()
