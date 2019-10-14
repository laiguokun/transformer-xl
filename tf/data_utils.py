"""Common data utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer("vocab_size", default=None, help="")
flags.DEFINE_integer("unk_id", default=None, help="")
flags.DEFINE_integer("bos_id", default=None, help="")
flags.DEFINE_integer("eos_id", default=None, help="")
flags.DEFINE_integer("cls_id", default=None, help="")
flags.DEFINE_integer("sep_id", default=None, help="")
flags.DEFINE_integer("pad_id", default=None, help="")
flags.DEFINE_integer("mask_id", default=None, help="")
flags.DEFINE_integer("eod_id", default=None, help="")
flags.DEFINE_integer("eop_id", default=None, help="")


default_symbols_mapping = collections.OrderedDict([
    ("<unk>", "unk_id"),
    ("<s>", "bos_id"),
    ("</s>", "eos_id"),
    ("<cls>", "cls_id"),
    ("<sep>", "sep_id"),
    ("<pad>", "pad_id"),
    ("<mask>", "mask_id"),
    ("<eod>", "eod_id"),
    ("<eop>", "eop_id")
])


def setup_special_ids(tokenizer, symbols_mapping=None):
  """Set up special ids."""
  FLAGS.vocab_size = tokenizer.get_vocab_size()
  tf.logging.info("Set vocab_size: %d.", FLAGS.vocab_size)

  if symbols_mapping is None:
    symbols_mapping = default_symbols_mapping
  for sym, sym_id_str in symbols_mapping.items():
    try:
      sym_id = tokenizer.get_token_id(sym)
      setattr(FLAGS, sym_id_str, sym_id)
      setattr(tokenizer, sym_id_str, sym_id)
      tf.logging.info("Set %s to %d.", sym_id_str, sym_id)
    except KeyError:
      tf.logging.warning("Skip %s: not found in tokenizer's vocab.", sym)


def format_filename(prefix, suffix, bsz_per_host, seq_len, lower_case=False):
  """docs."""
  if lower_case:
    case_str = "lower."
  else:
    case_str = ""

  file_name = "{}.lm.seq-{}.bsz-{}.{}{}".format(
      prefix, seq_len, bsz_per_host, case_str, suffix)

  return file_name


def convert_example(example, use_bfloat16=False):
  """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    if val.dtype == tf.int64:
      val = tf.cast(val, tf.int32)
    if use_bfloat16 and val.dtype == tf.float32:
      val = tf.cast(val, tf.bfloat16)

    example[key] = val
