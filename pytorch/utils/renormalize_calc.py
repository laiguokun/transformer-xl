from collections import Counter, OrderedDict

import torch
import torch.nn.functional as F


class RenormalizeVocabPost(object):
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
        self.device = torch.device("cpu")

    def set_cuda(self, cuda):
        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.head_vocab = self.head_vocab.to(self.device)
        self.end_vocab = self.end_vocab.to(self.device)

    def build_renormalize_vocab(self, ):
        self.lock = True
        self.load_vocab_mapping(self.vocab_map_fn)
        self.build_trie_tree()

    def build_trie_tree(self,):
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
                assert not (i>0 and self.head_vocab[token]==1), \
                    'middle token cannot be head token'
                node = node[token]
            for token in self.special_idx:
                node[token] = {}
        for token in self.special_idx:
            self.trie[token] = head_vocab_set

    def load_vocab_mapping(self, map_fn):
        self.head_vocab = torch.zeros(len(self.idx2sym)).byte()
        self.end_vocab = torch.zeros(len(self.idx2sym)).byte()
        self.vocab_map = {}
        with open(map_fn) as fin:
            for i, line in enumerate(fin):
                if self.lower_case:
                    line = line.lower()
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

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.lower_case:
                    line = line.lower()
                symb = line.strip().split()[0]
                self.add_symbol(symb)
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
        data = torch.from_numpy(data).to(self.device)
        target = torch.from_numpy(target).to(self.device)
        out_logit = torch.from_numpy(out_logit).to(self.device)

        next_word, new_status = self.get_next_word_set(data, last_status)
        head_vocab = self.head_vocab.reshape((1,1,-1))
        end_marks = self.get_end_marks(data).reshape(T, bsz, 1)
        head_vocab = head_vocab * end_marks
        next_word = next_word | head_vocab
        out_logit.masked_fill_(~next_word, -float('inf'))
        logit = -F.log_softmax(out_logit, dim=2)
        target = target.view(T, bsz, 1)
        nll = torch.gather(logit, 2, target)
        return nll, new_status

    def get_end_marks(self, symbols):
        symbols = symbols.view(-1,)
        ret = torch.gather(self.end_vocab, 0, symbols)
        return ret

    def get_next_word_set(self, symbols, last_status):
        next_word = torch.zeros(
            (symbols.shape[0], symbols.shape[1], len(self.sym2idx))).byte().to(self.device)
        new_status = []
        for batch_idx in range(symbols.shape[1]):
            node = last_status[batch_idx]
            for i in range(symbols.shape[0]):
                token = symbols[i][batch_idx].item()
                if self.head_vocab[token] == 1:
                    node = self.trie
                if node == -1:
                    next_word[i][batch_idx][:] = 1
                else:
                    token_list = torch.LongTensor(list(node[token].keys()))
                    next_word[i][batch_idx][token_list] = 1
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
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)