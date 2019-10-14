# coding=utf-8
"""Tokenization related."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain.data_utils import setup_special_ids
except ImportError as e:
  from data_utils import setup_special_ids
# pylint: enable=g-import-not-at-top


# Tokenizer related
flags.DEFINE_string("tokenizer_type", "whitespace",
                    help="Type of the tokenizer.")
flags.DEFINE_string("tokenizer_path", "",
                    help="Path to the tokenizer model, vocab, etc..")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased words.")

FLAGS = flags.FLAGS

SPIECE_UNDERLINE = "▁"


class WhitespaceTokenizer(object):
  """Whitespace separated tokenizer used for text."""

  def __init__(self, vocab_path, do_lower_case):
    self.do_lower_case = do_lower_case
    self._init_vocab(vocab_path)

  def _init_vocab(self, vocab_path):
    """Load vocab from text file."""
    self.idx2tok = []
    self.tok2idx = collections.OrderedDict()

    self.special_mapping = {}
    with tf.io.gfile.GFile(vocab_path, "r") as f:
      vocab = []
      for line in f:
        fields = line.strip().split()

        if fields:
          vocab.append(fields[0])
          if len(fields) == 2:
            self.special_mapping[fields[0]] = fields[1]

    for token in vocab:
      if token not in self.tok2idx:
        self.tok2idx[token] = len(self.idx2tok)
        self.idx2tok.append(token)

    for sym, sym_id_str in self.special_mapping.items():
      setattr(self, sym_id_str, self.tok2idx[sym])

  def get_token_id(self, token):
    return self.tok2idx.get(token, self.unk_id)

  def convert_text_to_tokens(self, text):
    """Converts a raw text into a list of tokens."""
    if self.do_lower_case:
      text = text.lower()
    return text.split()

  def convert_tokens_to_ids(self, tokens):
    """Converts a list of tokens into a list of int ids."""
    ids = [self.get_token_id(token) for token in tokens]
    return ids

  def convert_text_to_ids(self, text):
    tokens = self.convert_text_to_tokens(text)
    return self.convert_tokens_to_ids(tokens)

  def convert_ids_to_tokens(self, ids):
    """Converts a list of ids into a list of tokens."""
    tokens = [self.idx2tok[token_id] for token_id in ids]
    return tokens

  def convert_tokens_to_text(self, tokens):
    return " ".join(tokens)

  def convert_ids_to_text(self, ids):
    tokens = self.convert_ids_to_tokens(ids)
    return self.convert_tokens_to_text(tokens)

  def get_vocab_size(self):
    return len(self.idx2tok)


def get_tokenizer(**kwargs):
  """Initialize tokenizer."""
  if FLAGS.tokenizer_type == "whitespace":
    tokenizer = WhitespaceTokenizer(
        FLAGS.tokenizer_path, FLAGS.uncased, **kwargs)
  else:
    raise NotImplementedError

  setup_special_ids(tokenizer, tokenizer.special_mapping)

  return tokenizer
