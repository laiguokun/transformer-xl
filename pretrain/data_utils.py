"""Common data utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import multiprocessing

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np
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


def format_filename(prefix, suffix, uncased=False):
  """docs."""
  if not uncased:
    case_str = ""
  else:
    case_str = "uncased."

  file_name = "{}.{}{}".format(
      prefix, case_str, suffix)

  return file_name


def sparse_to_dense(example):
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    example[key] = val

  return example


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

  return example


def cpu_count():
  """Return the number of available cores."""
  num_available_cores = multiprocessing.cpu_count()
  return num_available_cores


def read_sents(file_path, tokenizer):
  """Read sentences from a file without order or doc structure."""
  all_sents = []
  tf.logging.info("===== Processing %s =====", file_path)
  for line in tf.gfile.Open(file_path):
    if len(all_sents) % 100000 == 0:
      tf.logging.info("Loading line %d", len(all_sents))

    cur_sent = tokenizer.convert_text_to_ids(line.strip())

    if FLAGS.add_double_eos:
      cur_sent = [tokenizer.eos_id] + cur_sent + [tokenizer.eos_id]
    elif FLAGS.add_eos:
      cur_sent = cur_sent + [tokenizer.eos_id]

    if not cur_sent:
      tf.logging.info("Skip line %s --> tokens %s", line.strip(), cur_sent)
      continue

    all_sents.append(np.array(cur_sent))

  tf.logging.info("Finish %s with %d lines", file_path, len(all_sents))

  return all_sents


def read_docs(file_path, tokenizer):
  """Read docs from a file separated by empty lines."""
  # working structure used to store each document
  all_docs = []
  doc, end_of_doc = [], False

  line_cnt = 0
  tf.logging.info("Start processing %s", file_path)
  for line in tf.gfile.Open(file_path):
    if line_cnt % 100000 == 0:
      tf.logging.info("Loading line %d", line_cnt)

    if not line.strip():
      # encounter an empty line (end of a document)
      end_of_doc = True
      cur_sent = []
    else:
      cur_sent = tokenizer.convert_text_to_ids(line.strip())

      if FLAGS.add_double_eos:
        cur_sent = [tokenizer.eos_id] + cur_sent + [tokenizer.eos_id]
      elif FLAGS.add_eos:
        cur_sent += [tokenizer.eos_id]

    if cur_sent:
      line_cnt += 1
      doc.append(np.array(cur_sent))

    # form a doc
    if end_of_doc:
      # only retain docs longer than `min_doc_len`
      doc_len = sum(map(len, doc))
      if doc_len >= max(FLAGS.min_doc_len, 1):
        all_docs.append(doc)

      # refresh working structs
      doc, end_of_doc = [], False

  # deal with the leafover if any
  if doc:
    # only retain docs longer than `min_doc_len`
    doc_len = sum(map(len, doc))
    if doc_len >= max(FLAGS.min_doc_len, 1):
      all_docs.append(doc)

  tf.logging.info("Finish %s with %d docs from %d lines.", file_path,
                  len(all_docs), line_cnt)

  return all_docs

