import os
from collections import Counter, OrderedDict

import torch

class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None, renormalize=False):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.renormalize = renormalize

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        print('build vocab from {}'.format(vocab_file))
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        if ('<UNK>' in self.sym2idx):
            self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def build_renormalize_vocab(self, path):
        print('loading renormalizing vocab')
        fn = os.path.join(path, 'head-end-vocab.pt')
        v = torch.load(fn)
        head_vocab = v['head_vocab']
        end_vocab = v['end_vocab']
        head_vocab = self.convert_to_tensor(head_vocab)
        end_vocab = self.convert_to_tensor(end_vocab)
        self.head_vocab = self.convert_to_idx_tensor(head_vocab)
        self.end_vocab = self.convert_to_idx_tensor(end_vocab)
        if ('<eos>' in self.sym2idx):
            self.end_vocab[self.sym2idx['<eos>']] = 1
            self.head_vocab[self.sym2idx['<eos>']] = 1

    def convert_to_idx_tensor(self, idx):
        ret = torch.zeros(len(self.idx2sym)).long()
        ret[idx] = 1
        return ret
        
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))
        if ordered:
            encoded = torch.cat(encoded)
        return encoded

    def encode_file_renormalize(self, path, path_next_word, ordered=False, verbose=False, 
            add_eos=True, add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded_symbols = []
        encoded_candidates = []
        encoded_end_marks = []
        with open(path, 'r', encoding='utf-8') as f:
            next_word_set = torch.load(path_next_word)
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos)
                symbols = self.convert_to_tensor(symbols)
                candidates =  [self.convert_to_tensor(tokens)
                                    for tokens in next_word_set[idx]]
                if add_eos:
                    candidates.append([])
                assert symbols.size(0) == len(candidates)
                end_mark = self.get_end_marks(symbols)
                #end_cnt += end_mark.sum()
                #tot_cnt += symbols.size(0)
                encoded_symbols.append(symbols)
                encoded_candidates.append(candidates)
                encoded_end_marks.append(end_mark)
                #sanity-check: pass
                #assert self.sanity_check(symbols, candidates, end_mark)
        encoded = [encoded_symbols, encoded_candidates, encoded_end_marks]
        if ordered:
            encoded[0] = torch.cat(encoded[0])
            encoded[1] = [item for sublist in encoded_candidates for item in sublist]
            encoded[2] = torch.cat(encoded[2])
        #print(end_cnt.item() / tot_cnt)
        return encoded

    def get_end_marks(self, symbols):
        ret = torch.zeros_like(symbols).long()
        for idx in range(symbols.size(0)):
            if self.end_vocab[symbols[idx]] == 1:
                ret[idx] = 1
        return ret

    def sanity_check(self, symbols, candidates, end_mark):
        for i in range(symbols.size(0) - 1):
            token1 = symbols[i]
            token2 = symbols[i+1]
            if end_mark[i] == 1 and self.head_vocab[token2] == True:
                continue
            s = set(candidates[i].tolist())
            if token2.item() in s:
                continue
            print(end_mark[i])
            print(self.convert_to_sent(symbols))
            print(i, token2, s)
            return False
        return True

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

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

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
            print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

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
