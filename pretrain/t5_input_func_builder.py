"""Create t5 input function for TPUEstimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
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
  from google3.experimental.users.zihangd.pretrain.mlm_input_func_builder import create_target_mapping
  from google3.experimental.users.zihangd.pretrain.mlm_input_func_builder import discrepancy_correction
except ImportError as e:
  from data_utils import type_cast
  from data_utils import sparse_to_dense
  from mlm_input_func_builder import chunk_to_sequence
  from mlm_input_func_builder import create_target_mapping
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS

flags.DEFINE_bool("origin_pos", default=True,
                  help="Use the original enc position for the dec inputs.")

def _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len, num_predict):
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
  target_mask = tf.reduce_sum(masked_matrix, axis=0)
  is_masked = tf.cast(target_mask, tf.bool)

  segment_range = tf.cast(tf.range(1, tf.shape(candidate_matrix)[0] + 1),
                          dtype=candidate_matrix.dtype)
  segment_matrix = segment_range[:, None] * candidate_matrix
  segment_ids = tf.reduce_sum(segment_matrix * masked_matrix, axis=0)
  segment_ids = tf.cast(segment_ids, dtype=inputs.dtype)

  pos_mat = tf.cumsum(candidate_matrix, axis=1, exclusive=True)
  pos_seq = tf.reduce_sum(pos_mat * masked_matrix, axis=0)

  return is_masked, segment_ids, pos_seq


def _token_span_mask(inputs, tgt_len, num_predict):
  """Sample token spans as prediction targets."""
  mask_alpha = tgt_len / num_predict
  round_to_int = lambda x: tf.cast(x, tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(FLAGS.min_tok, FLAGS.max_tok + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)

  # +1 is denote that in decoder there will be a placeholder, in encoder we will
  # remove the last one, this also prevent the adjacent span
  span_lens = tf.random.categorical(
      logits=logits[None],
      num_samples=num_predict,
      dtype=tf.int64,
  )[0] + FLAGS.min_tok + 1

  # Sample the ratio [0.0, 1.0) of left context lengths
  span_lens_float = tf.cast(span_lens, tf.float32)
  left_ratio = tf.random.uniform(shape=[num_predict], minval=0.0, maxval=1.0)
  left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)
  left_ctx_len = round_to_int(left_ctx_len)

  # Compute the offset from left start to the right end
  right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

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

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def create_t5_target(example, seq_len, num_predict, use_bfloat16):
  """docs."""
  inputs = example["inputs"]
  type_id = example.pop("type_id")

  ph_idx_0 = FLAGS.ph_id_0
  # sample mask
  is_masked, segment_ids, pos_seq = _token_span_mask(
      inputs, seq_len, num_predict)
  
  # because decoder has one place-holder, remove the last token in each span
  masked_idx = tf.boolean_mask(tf.range(seq_len), is_masked)
  is_beginning = tf.equal(tf.boolean_mask(pos_seq, is_masked), 0)
  ending_idx_m = tf.boolean_mask(tf.range(num_predict), is_beginning)[1:] 
  ending_idx_m = tf.concat(
      [ending_idx_m, tf.ones(shape=[1], dtype=tf.int32) * tf.shape(masked_idx)[0]],
      axis=0) - 1
  ending_idx = tf.gather(masked_idx, ending_idx_m)
  padding = tf.zeros(shape=[tf.shape(ending_idx)[0]])

  is_masked = tf.tensor_scatter_nd_update(
      is_masked,
      indices=ending_idx[:, None],
      updates=tf.cast(padding, dtype=is_masked.dtype))
  segment_ids = tf.tensor_scatter_nd_update(
      segment_ids,
      indices=ending_idx[:, None],
      updates=tf.cast(padding, dtype=segment_ids.dtype))
  pos_seq = tf.tensor_scatter_nd_update(
      pos_seq,
      indices=ending_idx[:, None],
      updates=tf.cast(padding, dtype=pos_seq.dtype))

  
  # encoder inp
  masked_idx = tf.boolean_mask(tf.range(seq_len), is_masked)
  is_beginning = tf.equal(tf.boolean_mask(pos_seq, is_masked), 0)
  beginning_idx = tf.boolean_mask(masked_idx, is_beginning)
  non_masked = tf.logical_not(is_masked)
  segment_num = tf.shape(beginning_idx)[0]
  enc_seq_mask = tf.tensor_scatter_nd_update(
      non_masked,
      indices=beginning_idx[:, None],
      updates=tf.ones(shape=[segment_num], dtype=tf.bool))
  enc_seq = tf.boolean_mask(inputs, enc_seq_mask)
  enc_type = tf.boolean_mask(type_id, enc_seq_mask)
  
  new_beginning_idx = tf.gather(
      tf.cumsum(tf.cast(enc_seq_mask, dtype=tf.int64), exclusive=True),
      beginning_idx)
  ph_idx_val = tf.cast(tf.range(segment_num) + ph_idx_0, dtype=tf.int64)
  enc_seq = tf.tensor_scatter_nd_update(
      enc_seq, 
      indices=new_beginning_idx[:, None],
      updates=ph_idx_val)

  pad_len = seq_len - tf.shape(enc_seq)[0]
  enc_seq = tf.concat(
      [enc_seq, tf.zeros(shape=[pad_len], dtype=enc_seq.dtype)],
      axis=0)
  enc_seq = tf.reshape(enc_seq, [seq_len])
  enc_type = tf.concat(
      [enc_type, tf.zeros(shape=[pad_len], dtype=enc_type.dtype)],
      axis=0)
  enc_type = tf.reshape(enc_type, [seq_len])

  example["source"] = enc_seq
  example["source_segmentation"] = tf.cast(tf.not_equal(enc_seq, 0), dtype=tf.int64)
  example["source_position"] = tf.range(seq_len, dtype=tf.int64)
  example["source_type"] = enc_type
  
  # decoder inp and out
  # add one because we append a final place holder at the end
  num_predict = num_predict + 1
  mask = tf.boolean_mask(tf.ones([seq_len], dtype=tf.int64), is_masked)
  mask = mask + tf.cast(is_beginning, mask.dtype)
  mask_idx = tf.cumsum(mask) - 1
  masked_tok = tf.boolean_mask(inputs, is_masked)
  dec_seq = tf.scatter_nd(
      shape=[num_predict],
      indices=mask_idx[:, None],
      updates=masked_tok)
  masked_type = tf.boolean_mask(type_id, is_masked)
  dec_type = tf.scatter_nd(
      shape=[num_predict],
      indices=mask_idx[:, None],
      updates=masked_type)

  ph_idx = tf.boolean_mask(mask_idx, is_beginning) - 1
  last_idx = mask_idx[-1] + 1
  ph_idx = tf.concat(
      [ph_idx, tf.ones(shape=[1], dtype=tf.int64) * last_idx],
      axis=0)
  ph_idx_val = tf.cast(tf.range(segment_num + 1) + ph_idx_0, dtype=tf.int64)
  dec_seq = tf.tensor_scatter_nd_update(
      dec_seq,
      indices=ph_idx[:, None],
      updates=ph_idx_val)
  
  pad_len = num_predict - (last_idx + 1)
  target_seg = tf.concat(
      [tf.ones(shape=[num_predict-pad_len], dtype=tf.int64),
       tf.zeros(shape=[pad_len], dtype=tf.int64)],
      axis=0
  )
  example["target"] = dec_seq
  example["target_segmentation"] = target_seg
  example["target_position"] = tf.range(num_predict, dtype=tf.int64)
  example["target_type"] = dec_type
  
  # type cast for example
  type_cast(example, use_bfloat16)

  for k, v in example.items():
    tf.logging.info("%s: %s", k, v)

  return example


def t5_process(dataset, seq_len, num_predict, use_bfloat16):
  """Process input tfrecords into proper format for t5 training."""
  dataset = chunk_to_sequence(dataset, seq_len)

  # Create t5 target
  create_t5_target_mapper = functools.partial(
      create_t5_target,
      seq_len=seq_len,
      num_predict=num_predict,
      use_bfloat16=use_bfloat16)
  dataset = dataset.map(create_t5_target_mapper, num_parallel_calls=64)

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


def sent_t5_dataset(params,
                      file_names,
                      num_hosts,
                      num_core_per_host,
                      seq_len,
                      num_predict,
                      is_training,
                      use_bfloat16=False,
                      num_threads=64,
                      record_shuffle_size=4096,
                      sequence_shuffle_size=2048):
  """Get sentence level t5 dataset."""
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
  dataset = t5_process(dataset, seq_len, num_predict, use_bfloat16)

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


def semidoc_t5_dataset(params,
                         file_names,
                         num_hosts,
                         num_core_per_host,
                         seq_len,
                         num_predict,
                         is_training,
                         use_bfloat16=False,
                         num_threads=64,
                         record_shuffle_size=256,
                         sequence_shuffle_size=2048):
  # pylint: disable=g-doc-args
  """Get semi-doc level t5 dataset.

  Notes:
  - Each sequence comes from the same document (except for boundary cases).
    This is different from the standard sent-level t5 dataset.
  - No consecutivity is ensured across batches, which is different from the
    standard doc-level t5 dataset.
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
  dataset = t5_process(dataset, seq_len, num_predict, use_bfloat16)

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


def doc_t5_dataset(params,
                     file_names,
                     num_hosts,
                     num_core_per_host,
                     seq_len,
                     num_predict,
                     is_training,
                     use_bfloat16=False,
                     num_threads=64,
                     record_shuffle_size=256):
  """Get document level t5 dataset."""

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
  process_shard = functools.partial(t5_process,
                                    seq_len=seq_len,
                                    num_predict=num_predict,
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
    num_predict,
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
  func_list = [doc_t5_dataset, semidoc_t5_dataset, sent_t5_dataset]

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
            num_predict=num_predict,
            use_bfloat16=use_bfloat16,
            **kwargs)

        datasets.append(cur_dataset)

    if len(datasets) > 1:
      dataset = tf.data.experimental.sample_from_datasets(datasets)
    elif len(datasets) == 1:
      dataset = datasets[0]

    return dataset

  return input_fn

