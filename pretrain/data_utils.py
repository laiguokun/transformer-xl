"""Common data utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

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

flags.DEFINE_integer("seg_id_a", default=0, help="segment id of segment A.")
flags.DEFINE_integer("seg_id_b", default=1, help="segment id of segment B.")
flags.DEFINE_integer("seg_id_cls", default=0, help="segment id of cls.")
flags.DEFINE_integer("seg_id_pad", default=0, help="segment id of pad.")

FLAGS = flags.FLAGS


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

      if sym_id_str != "unk_id" and sym_id == FLAGS.unk_id:
        tf.logging.warning("Skip %s: not found in tokenizer's vocab.", sym)
        continue

      setattr(FLAGS, sym_id_str, sym_id)
      tf.logging.info("Set %s to %d.", sym_id_str, sym_id)

    except KeyError:
      tf.logging.warning("Skip %s: not found in tokenizer's vocab.", sym)


def format_filename(prefix, suffix, seq_len, uncased=False):
  """docs."""
  if not uncased:
    case_str = ""
  else:
    case_str = "uncased."

  file_name = "{}.seq-{}.{}{}".format(
      prefix, seq_len, case_str, suffix)

  return file_name


def type_cast(example, use_bfloat16=False):
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
