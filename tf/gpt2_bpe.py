# coding=utf-8
"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT

This is the modified version from fairseq:
https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/gpt2_bpe_utils.py

DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

We change the `regex` to `re`
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import re

import tensorflow as tf


# @lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (list(range(ord(u"!"), ord(u"~") + 1)) +
          list(range(ord(u"¡"), ord(u"¬") + 1)) +
          list(range(ord(u"®"), ord(u"ÿ") + 1)))

    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    # cs = [chr(n) for n in cs]
    cs = [unichr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder(object):

    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        ##### Patterns use to extract complete "words" from text
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_ids = []

        # use regex to extract words
        words = re.findall(self.pat, text)
        for word in words:

            byte_word = ''.join(self.byte_encoder[ord(b)] for b in
                                word.encode('utf-8'))

            bpe_tokens = self.bpe(byte_word).split(' ')

            bpe_ids.extend(self.encoder[bpe_token] for bpe_token in bpe_tokens)

        return bpe_ids

    def decode(self, ids):
        # ids -> tokens
        tokens = [self.decoder[token_id] for token_id in ids]
        text = ''.join(tokens)
        # tokens -> string
        text = bytearray([self.byte_decoder[c] for c in text])
        return text.decode('utf-8', errors=self.errors)

def get_encoder(encoder_json_path, vocab_bpe_path):
    with tf.gfile.Open(encoder_json_path, 'r') as f:
        encoder = json.load(f)

    with tf.gfile.Open(vocab_bpe_path, 'r') as f:
        bpe_data = f.read()
    bpe_data = bpe_data.decode('utf-8')

    bpe_merges = [tuple(merge_str.split()) for merge_str in
                  bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

# if __name__ == '__main__':
#   HOME = '/usr/local/google/home/zihangd'
#   encoder = get_encoder('{}/data/gpt2bpe/encoder.json'.format(HOME),
#                         '{}/data/gpt2bpe/vocab.bpe'.format(HOME))
#   encoder.encode("Resistance")
