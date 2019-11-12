"""Create dae input function for TPUEstimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain.data_utils import type_cast
  from google3.experimental.users.zihangd.pretrain.data_utils import sparse_to_dense
  from google3.experimental.users.zihangd.pretrain.mlm_input_func_builder import chunk_to_sequence
except ImportError as e:
  from data_utils import type_cast
  from data_utils import sparse_to_dense
  from mlm_input_func_builder import chunk_to_sequence
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS

def _idx_pair_to_mask(
    beg_indices, 
    end_indices, 
    inputs, 
    tgt_len, 
    num_predict,
    del_rep_mask):
  """Turn begin and end indices into actual mask."""
  non_func_mask = tf.not_equal(inputs, FLAGS.eos_id)
  all_indices = tf.where(
      non_func_mask,
      tf.range(tgt_len, dtype=tf.int64),
      tf.constant(-1, shape=[tgt_len], dtype=tf.int64))
  candidate_matrix = tf.cast(
      tf.logical_and(
          all_indices[None, :] >= beg_indices[:, None],
          all_indices[None, :] < end_indices[:, None]),
      tf.float32)
  cumsum_matrix = tf.reshape(
      tf.cumsum(tf.reshape(candidate_matrix, [-1])),
      [-1, tgt_len])

  masked_matrix = (tf.cast(cumsum_matrix <= num_predict, tf.float32)
                   * candidate_matrix)
                   
  masked_type_matrix = (tf.cast(cumsum_matrix <= num_predict, tf.float32)
                        * (candidate_matrix * del_rep_mask[:, None]))
  is_masked = tf.reduce_sum(masked_type_matrix, axis=0)

  segment_range = tf.cast(tf.range(1, tf.shape(candidate_matrix)[0] + 1),
                          dtype=candidate_matrix.dtype)
  segment_matrix = segment_range[:, None] * candidate_matrix
  segment_ids = tf.reduce_sum(segment_matrix * masked_matrix, axis=0)
  segment_ids = tf.cast(segment_ids, dtype=inputs.dtype)

  pos_mat = tf.cumsum(candidate_matrix, axis=1, exclusive=True)
  pos_seq = tf.reduce_sum(pos_mat * masked_matrix, axis=0)

  return is_masked, segment_ids, pos_seq


def _token_span_mask(inputs, tgt_len, num_predict, del_rep_mask):
  """Sample token spans as prediction targets."""
  mask_alpha = tgt_len / num_predict
  round_to_int = lambda x: tf.cast(x, tf.int64)

  # Sample span lengths from a zipf distribution
  probs = np.array([1.0 for i in (FLAGS.min_tok, FLAGS.max_tok + 1)])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)

  # +1 is denote that in decoder there will be a placeholder, in encoder we will
  # remove the last one, this also prevent the adjacent span
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + FLAGS.min_tok

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_float = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len
  # remove the adjacent case
  left_ctx_len = left_ctx_len + 1

  # Get the actual begin and end indices
  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range indices
  valid_idx_mask = end_indices <= tgt_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random_shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)
  del_rep_mask = tf.gather(del_rep_mask, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict, del_rep_mask) 

def get_type_id(tgt_len, tgt_idx, type_val):
  """Deal with type id for insertion."""
  tgt_idx_left_shift = tgt_idx[:-1]
  type_val_right_shift = type_val[1:]
  new_type_id_shift = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx_left_shift[:, None],
      updates=type_val_right_shift
      )
  new_type_id_shift = tf.concat([type_val[:1], new_type_id_shift], axis=0)
  new_type_id_shift = tf.math.cumsum(new_type_id_shift, exclusive=True)[1:]
  new_type_id = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx[:, None],
      updates=type_val
      )
  new_type_id = tf.math.cumsum(new_type_id, exclusive=True)
  new_type_id = new_type_id_shift - new_type_id
  return new_type_id


def create_dae_features(example, seq_len, use_bfloat16):
  """Create features used for DAE seq2seq pretraining."""
  ##### original sequence of length `seq_len - 1`
  inputs = example.pop("inputs")
  tf_int = inputs.dtype

  # Append <eos>
  inputs = tf.concat(
      [inputs, tf.constant(FLAGS.eos_id, shape=[1], dtype=tf_int)], 0)
  eos_mask = tf.equal(inputs, FLAGS.eos_id)
  non_eos_mask = tf.logical_not(eos_mask)
  example["inputs"] = inputs

  type_id = example.pop("type_id")
  type_id = tf.concat([type_id, type_id[-1:]], 0)
  example["type_id"] = type_id

  num_predict = int(seq_len * (FLAGS.del_ratio + FLAGS.rep_ratio))
  # generate delete and replace mask for span
  alpha = FLAGS.del_ratio / (FLAGS.del_ratio + FLAGS.rep_ratio)
  uniform = tf.random.uniform(shape=[num_predict], minval=0, maxval=1)
  # delete and rep mask 1 for del and 2 for rep
  del_rep_mask = tf.cast(uniform < alpha, tf.float32) + 1

  del_rep_mask, segment_ids, pos_seq = _token_span_mask(
      inputs, seq_len, num_predict, del_rep_mask)
  
  del_mask = tf.equal(del_rep_mask, 1)
  non_del_mask = tf.logical_not(del_mask)
  rep_mask = tf.equal(del_rep_mask, 2)

  # maximum number of edit op
  ins_num = int(math.ceil(seq_len * FLAGS.ins_ratio))
  del_num = num_predict
  rep_num = num_predict
  actual_rep_num = tf.reduce_sum(tf.cast(rep_mask, tf.float32))
  actual_del_num = tf.reduce_sum(tf.cast(del_mask, tf.float32))

  # Sample insertion positions given deletion
  non_ins_mask = del_rep_mask > 0
  right_shift_del_sup_mask = tf.concat(
      [tf.constant(False, shape=[1], dtype=tf.bool), non_ins_mask[:-1]], axis=0)
  non_ins_mask = tf.logical_or(non_ins_mask, right_shift_del_sup_mask)
  non_ins_mask = tf.logical_or(non_ins_mask, eos_mask)
  # obtain a shuffled index
  ins_idx = tf.range(seq_len, dtype=tf.int64)
  ins_idx = tf.boolean_mask(ins_idx, tf.logical_not(non_ins_mask))
  ins_idx = tf.random_shuffle(ins_idx)
  
  # uniform span length
  logits = tf.ones(shape=[FLAGS.max_tok - FLAGS.min_tok + 1])
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=tf.shape(ins_idx)[0],
      dtype=tf_int,
  )[0] + FLAGS.min_tok
  valid_span = tf.cumsum(span_lens) < ins_num
  
  ins_idx = tf.boolean_mask(ins_idx, valid_span)
  ins_cnt = tf.boolean_mask(span_lens, valid_span)
  ins_cnt = tf.scatter_nd(
      shape=[seq_len], 
      indices=ins_idx[:, None],
      updates=ins_cnt
  )

  # change replaced locations to <mask>
  rep_input = tf.where(
      rep_mask,
      tf.constant(FLAGS.mask_id, shape=[seq_len], dtype=tf_int),
      inputs)

  ######### construct features
  ori_idx = tf.range(seq_len, dtype=tf_int)

  ############################
  ##### Encoder features #####
  ############################
  # map `ori_idx` to `enc_idx`
  enc_shift = ins_cnt - tf.cast(del_mask, tf_int)
  enc_shift = tf.cumsum(enc_shift)
  enc_shift_idx = ori_idx + enc_shift

  # exclude idx larger than `FLAGS.enc_len` and deleted positions
  enc_valid_mask = tf.logical_and(enc_shift_idx < FLAGS.enc_len, non_del_mask)
  enc_idx = tf.boolean_mask(enc_shift_idx, enc_valid_mask)
  enc_val = tf.boolean_mask(rep_input, enc_valid_mask)
  enc_type = tf.boolean_mask(type_id, enc_valid_mask)

  # encoder padding mask
  enc_non_pad_len = tf.math.reduce_max(enc_idx) + 1
  enc_pad_len = FLAGS.enc_len - enc_non_pad_len
  enc_mask = tf.concat(
      [tf.zeros(shape=[enc_non_pad_len], dtype=tf.float32),
       tf.ones(shape=[enc_pad_len], dtype=tf.float32)], 0)
  enc_mask.set_shape([FLAGS.enc_len])

  # (a) Map all non-padding positions to `mask_id`
  enc_seq = tf.concat(
      [tf.zeros(shape=[enc_non_pad_len], dtype=tf_int) + FLAGS.mask_id,
       tf.zeros(shape=[enc_pad_len], dtype=tf_int) + FLAGS.pad_id], 0)
  enc_seq.set_shape([FLAGS.enc_len])
  # (b) Scatter replaced input to proper positions: unscattered positions remain
  # to be `mask_id` for insertion
  enc_seq = tf.tensor_scatter_nd_update(
      enc_seq,
      indices=enc_idx[:, None],
      updates=enc_val)

  enc_type = get_type_id(FLAGS.enc_len, enc_idx, enc_type)

  ##############################
  ##### Generator features #####
  ##############################
  is_masked = tf.equal(enc_seq, FLAGS.mask_id)

  # gen_mask_map: permutation matrix used to extract `enc_num_mask` subset
  # from the `enc_len` full sequence
  enc_masked_idx = tf.boolean_mask(tf.range(FLAGS.enc_len, dtype=tf_int),
                                   is_masked)
  enc_num_mask = rep_num + ins_num
  actual_enc_num_mask = tf.shape(enc_masked_idx)[0]
  map_pad_len = enc_num_mask - actual_enc_num_mask
  gen_mask_map = tf.one_hot(enc_masked_idx, FLAGS.enc_len, dtype=tf.float32)
  map_padding = tf.zeros([map_pad_len, FLAGS.enc_len], dtype=gen_mask_map.dtype)
  gen_mask_map = tf.concat([gen_mask_map, map_padding], axis=0)
  gen_mask_map = tf.reshape(gen_mask_map, [enc_num_mask, FLAGS.enc_len])

  # gen_tgt_mask: only `rep_num` 1s in the sequence of length `enc_num_mask`
  no_ins_inp = tf.scatter_nd(shape=[FLAGS.enc_len],
                             indices=enc_idx[:, None],
                             updates=enc_val)
  is_rep = tf.equal(no_ins_inp, FLAGS.mask_id)
  gen_tgt_mask = tf.boolean_mask(is_rep, is_masked)
  gen_tgt_mask.set_shape([enc_num_mask])

  # gen_tgt: scatter `rep_num` replaced ids to the correct positions in the
  # sequence of total length `enc_num_mask` (others correspond to insertions)
  enc_valid_rep_mask = tf.logical_and(rep_mask, enc_shift_idx < FLAGS.enc_len)
  gen_tgt_val = tf.boolean_mask(inputs, enc_valid_rep_mask)
  gen_tgt_idx = tf.boolean_mask(tf.range(enc_num_mask, dtype=tf_int),
                                gen_tgt_mask)
  gen_tgt = tf.tensor_scatter_nd_update(
      tf.constant(FLAGS.pad_id, shape=[enc_num_mask], dtype=tf_int),
      indices=gen_tgt_idx[:, None],
      updates=gen_tgt_val)

  ############################
  ##### Decoder features #####
  ############################
  # We do not delete any token for decoder: the shift only comes from insertion
  dec_shift = tf.cumsum(ins_cnt)
  dec_idx = ori_idx + dec_shift

  # Similarlly, exclude idx larger than `FLAGS.dec_len`
  dec_valid_mask = dec_idx < FLAGS.dec_len
  dec_idx = tf.boolean_mask(dec_idx, dec_valid_mask)
  dec_val = tf.boolean_mask(inputs, dec_valid_mask)
  dec_type = tf.boolean_mask(type_id, dec_valid_mask)

  # decoder mask
  dec_non_pad_len = tf.math.reduce_max(dec_idx) + 1
  dec_pad_len = FLAGS.dec_len - dec_non_pad_len
  dec_mask = tf.concat(
      [tf.zeros(shape=[dec_non_pad_len], dtype=tf.float32),
       tf.ones(shape=[dec_pad_len], dtype=tf.float32)], 0)
  dec_mask.set_shape([FLAGS.dec_len])

  # (a) Map all non-padding positions to `ins_id`
  dec_seq = tf.concat(
      [tf.zeros(shape=[dec_non_pad_len], dtype=tf_int) + FLAGS.ins_id,
       tf.zeros(shape=[dec_pad_len], dtype=tf_int) + FLAGS.pad_id], 0)
  dec_seq.set_shape([FLAGS.dec_len])

  # (b) Scatter original input to proper positions: unscattered positions remain
  # to be `ins_id` for insertion
  dec_seq = tf.tensor_scatter_nd_update(
      dec_seq,
      indices=dec_idx[:, None],
      updates=dec_val)

  # decoder input
  dec_inp = tf.concat([tf.constant(FLAGS.eos_id, shape=[1], dtype=tf_int),
                       dec_seq[:-1]], 0)

  # decoder type
  dec_type = get_type_id(FLAGS.dec_len, dec_idx, dec_type)
  dec_type = tf.concat([dec_type[:1], dec_type[:-1]], 0)

  # edit type label
  dec_ins_mask = tf.equal(dec_seq, FLAGS.ins_id)
  dec_rep_mask = tf.scatter_nd(
      shape=[FLAGS.dec_len],
      indices=dec_idx[:, None],
      updates=tf.boolean_mask(rep_mask, dec_valid_mask))
  dec_del_mask = tf.scatter_nd(
      shape=[FLAGS.dec_len],
      indices=dec_idx[:, None],
      updates=tf.boolean_mask(del_mask, dec_valid_mask))
  edit_label = tf.cast(dec_ins_mask, tf_int) * FLAGS.ins_label
  edit_label += tf.cast(dec_rep_mask, tf_int) * FLAGS.rep_label
  edit_label += tf.cast(dec_del_mask, tf_int) * FLAGS.del_label

  # NOTE: only use delete and replace positions for decoder LM loss (no insert)
  # `dec_mask_map`: permutation matrix used to extract `dec_num_mask` subset
  #                 from the `dec_len` full sequence
  dec_edit_mask = tf.logical_or(dec_rep_mask, dec_del_mask)
  dec_edit_idx = tf.boolean_mask(tf.range(FLAGS.dec_len), dec_edit_mask)
  dec_num_mask = rep_num + del_num
  dec_actual_num_mask = tf.shape(dec_edit_idx)[0]
  map_pad_len = dec_num_mask - dec_actual_num_mask
  dec_mask_map = tf.one_hot(dec_edit_idx, FLAGS.dec_len, dtype=tf.float32)
  map_padding = tf.zeros([map_pad_len, FLAGS.dec_len], dtype=dec_mask_map.dtype)
  dec_mask_map = tf.concat([dec_mask_map, map_padding], axis=0)
  dec_mask_map = tf.reshape(dec_mask_map, [dec_num_mask, FLAGS.dec_len])

  # `dec_tgt_mask`: only first `dec_actual_num_mask` elements are 1s in the
  #                 `dec_tgt_mask` of total length `dec_num_mask`.
  dec_tgt_mask = tf.concat(
      [tf.ones([dec_actual_num_mask], dtype=tf.float32),
       tf.zeros([map_pad_len], dtype=tf.float32)],
      axis=0)
  dec_tgt_mask.set_shape([dec_num_mask])

  # dec_masked_tgt: the subset of target tokens used to LM loss
  dec_mask_val = tf.boolean_mask(dec_seq, dec_edit_mask)
  dec_mask_idx = tf.range(dec_actual_num_mask)
  dec_masked_tgt = tf.tensor_scatter_nd_update(
      tf.constant(FLAGS.pad_id, shape=[dec_num_mask], dtype=tf_int),
      indices=dec_mask_idx[:, None],
      updates=dec_mask_val)

  # Convert the rep idx from encoder part to decoder. Used to remove the loss
  # of generated tokens that are equal to original tokens

  if actual_rep_num > 0:
    # `rep_enc2dec_full`: permutation matrix [enc_num_mask, dec_len]
    enc_rep_map_idx = tf.boolean_mask(tf.range(enc_num_mask), gen_tgt_mask)
    dec_rep_idx = tf.boolean_mask(tf.range(FLAGS.dec_len), dec_rep_mask)
    dec_rep_idx = tf.one_hot(dec_rep_idx, FLAGS.dec_len)
    rep_enc2dec_full = tf.scatter_nd(
        shape=[enc_num_mask, FLAGS.dec_len],
        indices=enc_rep_map_idx[:, None],
        updates=dec_rep_idx)
    rep_enc2dec_full = tf.cast(rep_enc2dec_full, tf.float32)

    # `rep_enc2dec_part`: permutation matrix [enc_num_mask, dec_num_mask]
    is_rep = tf.boolean_mask(dec_rep_mask, dec_edit_mask)
    dec_rep_map_idx = tf.boolean_mask(tf.range(dec_num_mask), is_rep)
    dec_rep_map_idx = tf.one_hot(dec_rep_map_idx, dec_num_mask)
    rep_enc2dec_part = tf.scatter_nd(
        shape=[enc_num_mask, dec_num_mask],
        indices=enc_rep_map_idx[:, None],
        updates=dec_rep_map_idx)
    rep_enc2dec_part = tf.cast(rep_enc2dec_part, tf.float32)
  else:
    rep_enc2dec_full = tf.zeros(shape=[enc_num_mask, FLAGS.dec_len])
    rep_enc2dec_part = tf.zeros(shape=[enc_num_mask, dec_num_mask])


  ##### Put everything into the example
  example["gen_inp"] = enc_seq
  example["gen_tgt"] = gen_tgt
  example["gen_mask_map"] = gen_mask_map
  example["gen_tgt_mask"] = tf.cast(gen_tgt_mask, tf.float32)

  example["enc_type"] = enc_type
  example["enc_mask"] = enc_mask

  example["dec_inp"] = dec_inp
  example["dec_tgt"] = dec_seq
  example["dec_type"] = dec_type
  example["dec_mask"] = dec_mask
  example["edit_label"] = edit_label
  example["dec_mask_map"] = dec_mask_map
  example["dec_masked_tgt"] = dec_masked_tgt
  example["dec_lm_tgt_mask"] = dec_tgt_mask
  example["rep_enc2dec_full"] = rep_enc2dec_full
  example["rep_enc2dec_part"] = rep_enc2dec_part

  ##### type cast for example
  type_cast(example, use_bfloat16)
  for k, v in example.items():
    tf.logging.info("%s: %s", k, v)

  return example


def dae_process(dataset, seq_len, use_bfloat16):
  """Process input tfrecords into proper format for dae training."""
  # `- 1` to account for <eos>
  dataset = chunk_to_sequence(dataset, seq_len - 1)

  # Create dae target
  dae_features_mapper = functools.partial(
      create_dae_features,
      seq_len=seq_len,
      use_bfloat16=use_bfloat16)
  dataset = dataset.map(dae_features_mapper, num_parallel_calls=64)

  return dataset


def get_record_parser():
  """Config tfrecord parser."""
  def parser(record):
    """function used to parse tfrecord."""

    record_spec = {
        "inputs": tf.VarLenFeature(tf.int64),
        "type_id": tf.FixedLenFeature([1], tf.int64),
    }

    # retrieve serialized example
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    inputs = example["inputs"]
    inp_len = tf.shape(inputs)[0]

    # expand type id to full length
    example["type_id"] = tf.broadcast_to(example["type_id"], [inp_len])

    # convert all sparse example to dense
    example = sparse_to_dense(example)

    return example

  return parser


def parse_record(dataset,
                 parser,
                 is_training,
                 num_threads=64,
                 file_shuffle_size=None,
                 record_shuffle_size=None):
  """Parse tfrecords in a dataset."""

  if is_training:
    # file-level shuffle
    if file_shuffle_size and file_shuffle_size > 1:
      tf.logging.info("File level shuffle with size %d", file_shuffle_size)
      dataset = dataset.shuffle(file_shuffle_size)

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(8, file_shuffle_size)
    tf.logging.info("Interleave %d files", cycle_length)

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=True,
            cycle_length=cycle_length))

    if record_shuffle_size and record_shuffle_size > 1:
      tf.logging.info("Record level shuffle with size %d", record_shuffle_size)
      dataset = dataset.shuffle(buffer_size=record_shuffle_size)

    dataset = dataset.map(parser, num_parallel_calls=num_threads)
    dataset = dataset.cache().repeat()
  else:
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(parser)

  return dataset


def sent_dae_dataset(params,
                     file_names,
                     num_hosts,
                     num_core_per_host,
                     seq_len,
                     is_training,
                     use_bfloat16=False,
                     num_threads=64,
                     record_shuffle_size=4096,
                     sequence_shuffle_size=2048):
  """Get sentence level dae dataset."""
  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  ##### Split input files across hosts
  if len(file_names) >= num_hosts:
    file_paths = file_names[host_id::num_hosts]
  else:
    file_paths = file_names
  tf.logging.info("Host %d handles %d files:", host_id, len(file_paths))

  ##### Parse records
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)
  dataset = parse_record(dataset=dataset,
                         parser=get_record_parser(),
                         is_training=is_training,
                         num_threads=num_threads,
                         file_shuffle_size=len(file_paths),
                         record_shuffle_size=record_shuffle_size)

  # process dataset
  dataset = dae_process(dataset, seq_len, use_bfloat16)

  # Sequence level shuffle
  if is_training and sequence_shuffle_size:
    tf.logging.info("Seqeunce level shuffle with size %d",
                    sequence_shuffle_size)
    dataset = dataset.shuffle(buffer_size=sequence_shuffle_size)

  # batching
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)

  # Prefetch
  dataset = dataset.prefetch(num_core_per_host)

  return dataset


def semidoc_dae_dataset(params,
                        file_names,
                        num_hosts,
                        num_core_per_host,
                        seq_len,
                        is_training,
                        use_bfloat16=False,
                        num_threads=64,
                        record_shuffle_size=256,
                        sequence_shuffle_size=2048):
  # pylint: disable=g-doc-args
  """Get semi-doc level dae dataset.

  Notes:
  - Each sequence comes from the same document (except for boundary cases).
    This is different from the standard sent-level dae dataset.
  - No consecutivity is ensured across batches, which is different from the
    standard doc-level dae dataset.
  - Effectively, semi-doc dataset maintains short range (seq_len) dependency,
    which is more random than doc-level and less random than sent-level.

  Returns:
    a tf.data.Dataset
  """
  # pylint: enable=g-doc-args
  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  ##### Split input files across hosts
  if len(file_names) >= num_hosts:
    file_paths = file_names[host_id::num_hosts]
  else:
    file_paths = file_names
  tf.logging.info("Host %d handles %d files:", host_id, len(file_paths))

  ##### Parse records
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)
  dataset = parse_record(dataset=dataset,
                         parser=get_record_parser(),
                         is_training=is_training,
                         num_threads=num_threads,
                         file_shuffle_size=len(file_paths),
                         record_shuffle_size=record_shuffle_size)

  # process dataset
  dataset = dae_process(dataset, seq_len, use_bfloat16)

  # Sequence level shuffle
  if is_training and sequence_shuffle_size:
    tf.logging.info("Seqeunce level shuffle with size %d",
                    sequence_shuffle_size)
    dataset = dataset.shuffle(buffer_size=sequence_shuffle_size)

  # batching
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)

  # Prefetch
  dataset = dataset.prefetch(num_core_per_host)

  return dataset


def doc_dae_dataset(params,
                    file_names,
                    num_hosts,
                    num_core_per_host,
                    seq_len,
                    is_training,
                    use_bfloat16=False,
                    num_threads=64,
                    record_shuffle_size=256):
  """Get document level dae dataset."""

  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  ##### Split input files across hosts
  if len(file_names) >= num_hosts:
    file_paths = file_names[host_id::num_hosts]
  else:
    file_paths = file_names
  tf.logging.info("Host %d handles %d files:", host_id, len(file_paths))

  ##### Create dataset from file_paths
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  if len(file_paths) // bsz_per_core >= 2:
    ##### Enough input files, so do file-level sharding shard

    # Split the dataset into `bsz_per_core` disjoint shards
    shards = [dataset.shard(bsz_per_core, i) for i in range(bsz_per_core)]

    # Parse records
    file_shuffle_size = (len(file_paths) + bsz_per_core - 1) // bsz_per_core
    parse_shard = functools.partial(
        parse_record,
        parser=get_record_parser(),
        is_training=is_training,
        num_threads=num_threads,
        file_shuffle_size=file_shuffle_size,
        record_shuffle_size=record_shuffle_size)
    shards = [parse_shard(dataset=shard) for shard in shards]
  else:
    ##### Not enough input files, so do record-level sharding

    # Parse records
    dataset = parse_record(dataset,
                           parser=get_record_parser(),
                           is_training=is_training,
                           num_threads=num_threads,
                           file_shuffle_size=len(file_names),
                           record_shuffle_size=record_shuffle_size)

    # Split the dataset into `bsz_per_core` disjoint shards
    shards = [dataset.shard(bsz_per_core, i) for i in range(bsz_per_core)]

  # process each shard
  process_shard = functools.partial(dae_process,
                                    seq_len=seq_len,
                                    use_bfloat16=use_bfloat16)
  shards = [process_shard(dataset=shard) for shard in shards]

  # merge shards into a single batched dataset
  def batch_zipped_dataset(*features):
    """Stack a list of homogeneous inputs from a zipped dataset into one."""
    new_feature = {}
    for key in features[0].keys():
      tensor_list = [f[key] for f in features]
      new_feature[key] = tf.stack(tensor_list, axis=0)  # [sum bsz, length]
    return new_feature
  dataset = tf.data.Dataset.zip(tuple(shards))
  dataset = dataset.map(batch_zipped_dataset)

  # Prefetch
  dataset = dataset.prefetch(num_core_per_host)

  return dataset


def get_input_fn(
    doc_dir,
    semi_dir,
    sent_dir,
    split,
    uncased,
    seq_len,
    bsz_per_host,
    num_hosts=1,
    num_core_per_host=1,
    use_bfloat16=False,
    **kwargs):
  """Create Estimator input function."""

  def dir_to_paths(data_dir, data_type):
    """Get data file paths in the given dir."""
    file_paths = []

    if data_dir:
      tf.logging.info("=" * 120)

      case_str = "uncased." if uncased else ""
      glob_base = "data.{}.{}.{}tfrecord*".format(split, data_type, case_str)

      for idx, dir_path in enumerate(data_dir.split(",")):
        glob = os.path.join(dir_path, glob_base)
        cur_file_paths = sorted(tf.io.gfile.glob(glob))
        file_paths += cur_file_paths

        tf.logging.info("[%d] Data glob: %s", idx, glob)
        tf.logging.info("[%d] Num of file path: %d", idx, len(cur_file_paths))

      tf.logging.info("[%s] Total number of file path: %d", data_type,
                      len(file_paths))

    return file_paths

  doc_files = dir_to_paths(doc_dir, "doc")
  semi_files = dir_to_paths(semi_dir, "doc")
  sent_files = dir_to_paths(sent_dir, "sent")

  file_list = [doc_files, semi_files, sent_files]
  func_list = [doc_dae_dataset, semidoc_dae_dataset, sent_dae_dataset]

  def input_fn(params):
    """Construct input function for TPUEstimator."""
    assert params["batch_size"] * num_core_per_host == bsz_per_host

    datasets = []
    for files, func in zip(file_list, func_list):
      if files:
        cur_dataset = func(
            params=params,
            num_hosts=num_hosts,
            num_core_per_host=num_core_per_host,
            is_training=split == "train",
            file_names=files,
            seq_len=seq_len,
            use_bfloat16=use_bfloat16,
            **kwargs)

        datasets.append(cur_dataset)

    if len(datasets) > 1:
      dataset = tf.data.experimental.sample_from_datasets(datasets)
    elif len(datasets) == 1:
      dataset = datasets[0]

    return dataset

  return input_fn

