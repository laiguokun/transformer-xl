"""Create MLM input function for TPUEstimator."""
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
except ImportError as e:
  from data_utils import type_cast
  from data_utils import sparse_to_dense
# pylint: enable=g-import-not-at-top

FLAGS = flags.FLAGS
flags.DEFINE_enum("sample_strategy", default="single_token",
                  enum_values=["single_token", "token_span"],
                  help="Stragey used to sample prediction targets.")

flags.DEFINE_float("leak_ratio", default=0.1,
                   help="Percent of masked positions that are filled with "
                   "original tokens.")
flags.DEFINE_float("rand_ratio", default=0.1,
                   help="Percent of masked positions that are filled with "
                   "random tokens.")

flags.DEFINE_integer("max_tok", default=4,
                     help="Maximum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")
flags.DEFINE_integer("min_tok", default=2,
                     help="Minimum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")

flags.DEFINE_bool("add_eos", False,
                  help="whether to append EOS at the end of a line.")
flags.DEFINE_bool("add_double_eos", False,
                  help="whether to append EOS at the begin and end of a line.")


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
  masked_matrix = tf.cast(cumsum_matrix <= num_predict, tf.float32)
  target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
  is_masked = tf.cast(target_mask, tf.bool)

  return is_masked


def _token_span_mask(inputs, tgt_len, num_predict):
  """Sample token spans as prediction targets."""
  mask_alpha = tgt_len / num_predict
  round_to_int = lambda x: tf.cast(tf.round(x), tf.int64)

  # Sample span lengths from a zipf distribution
  span_len_seq = np.arange(FLAGS.min_tok, FLAGS.max_tok + 1)
  probs = np.array([1.0 /  (i + 1) for i in span_len_seq])

  probs /= np.sum(probs)
  logits = tf.constant(np.log(probs), dtype=tf.float32)
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

  # Get the actual begin and end indices
  beg_indices = (tf.cumsum(left_ctx_len) +
                 tf.cumsum(right_offset, exclusive=True))
  end_indices = beg_indices + span_lens

  # Remove out of range indices
  valid_idx_mask = end_indices < tgt_len
  beg_indices = tf.boolean_mask(beg_indices, valid_idx_mask)
  end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

  # Shuffle valid indices
  num_valid = tf.cast(tf.shape(beg_indices)[0], tf.int64)
  order = tf.random_shuffle(tf.range(num_valid, dtype=tf.int64))
  beg_indices = tf.gather(beg_indices, order)
  end_indices = tf.gather(end_indices, order)

  return _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len,
                           num_predict)


def _single_token_mask(inputs, tgt_len, num_predict, exclude_mask=None):
  """Sample individual tokens as prediction targets."""
  func_mask = tf.equal(inputs, FLAGS.eos_id)
  if exclude_mask is None:
    exclude_mask = func_mask
  else:
    exclude_mask = tf.logical_or(func_mask, exclude_mask)
  candidate_mask = tf.logical_not(exclude_mask)

  all_indices = tf.range(tgt_len, dtype=tf.int64)
  candidate_indices = tf.boolean_mask(all_indices, candidate_mask)
  masked_pos = tf.random_shuffle(candidate_indices)
  masked_pos = tf.contrib.framework.sort(masked_pos[:num_predict])
  target_mask = tf.sparse_to_dense(
      sparse_indices=masked_pos,
      output_shape=[tgt_len],
      sparse_values=1.0,
      default_value=0.0)
  is_masked = tf.cast(target_mask, tf.bool)

  return is_masked


def online_sample_masks(inputs, tgt_len, num_predict):
  """Sample target positions to predict."""
  tf.logging.info("Online sample with strategy: `%s`.", FLAGS.sample_strategy)
  if FLAGS.sample_strategy == "single_token":
    return _single_token_mask(inputs, tgt_len, num_predict)
  elif FLAGS.sample_strategy == "token_span":
    is_masked = _token_span_mask(inputs, tgt_len, num_predict)
    cur_num_masked = tf.reduce_sum(tf.cast(is_masked, tf.int64))
    extra_mask = _single_token_mask(
        inputs, tgt_len, num_predict - cur_num_masked, is_masked)
    return tf.logical_or(is_masked, extra_mask)
  else:
    raise NotImplementedError


def discrepancy_correction(inputs, is_masked, tgt_len):
  """Construct the masked input."""
  random_p = tf.random.uniform([tgt_len], maxval=1.0)
  mask_ids = tf.constant(FLAGS.mask_id, dtype=inputs.dtype, shape=[tgt_len])

  change_to_mask = tf.logical_and(random_p > FLAGS.leak_ratio, is_masked)
  masked_ids = tf.where(change_to_mask, mask_ids, inputs)

  if FLAGS.rand_ratio > 0:
    change_to_rand = tf.logical_and(
        FLAGS.leak_ratio < random_p,
        random_p < FLAGS.leak_ratio + FLAGS.rand_ratio)
    change_to_rand = tf.logical_and(change_to_rand, is_masked)
    rand_ids = tf.random.uniform([tgt_len], maxval=FLAGS.vocab_size,
                                 dtype=masked_ids.dtype)
    masked_ids = tf.where(change_to_rand, rand_ids, masked_ids)

  return masked_ids


def create_perm(inputs, is_masked, seq_len):
  """Sample a factorization order, and create an attention mask accordingly.

  Args:
    inputs: int64 Tensor in shape [seq_len], input ids.
    is_masked: bool Tensor in shape [seq_len]. True means being selected
      for partial prediction.
    seq_len: int, sequence length.

  Returns:
    perm_mask_k: flat32 Tensor in shape [seq_len, seq_len].
    perm_mask_q: flat32 Tensor in shape [seq_len, seq_len].
    inputs_k: int64 Tensor in shape [seq_len], input ids.
    inputs_q: int64 Tensor in shape [seq_len], masked input ids.
    is_masekd: bool Tensor in shape [seq_len], whether a target
  """

  # Generate permutation indices
  index = tf.range(seq_len, dtype=tf.int64)
  index = tf.random_shuffle(index)

  # non-functional tokens
  non_func_tokens = tf.not_equal(inputs, FLAGS.eos_id)
  masked_tokens = tf.logical_and(is_masked, non_func_tokens)
  non_masked_or_func_tokens = tf.logical_not(masked_tokens)

  # Randomly leak some masked tokens
  random_p = tf.random.uniform([seq_len], maxval=1.0)
  if FLAGS.leak_ratio > 0:
    leak_tokens = tf.logical_and(
        masked_tokens,
        random_p < FLAGS.leak_ratio)
    can_attend_self = tf.logical_or(non_masked_or_func_tokens, leak_tokens)
  else:
    can_attend_self = non_masked_or_func_tokens

  ##### perm_mask_k & perm_mask_q
  smallest_index = -2 * tf.ones([seq_len], dtype=tf.int64)
  to_index = tf.where(can_attend_self, smallest_index, index)
  from_index = tf.where(can_attend_self, to_index + 1, to_index)

  # For masked tokens, can attend if i > j
  # For context tokens, always can attend each other
  can_attend_q = from_index[:, None] > to_index[None, :]
  can_attend_k = from_index[:, None] >= to_index[None, :]

  # In modeling, 1 indicates cannot attend. Hence, reverse the value here.
  perm_mask_q = 1.0 - tf.cast(can_attend_q, tf.float32)
  perm_mask_k = 1.0 - tf.cast(can_attend_k, tf.float32)

  ##### inputs_k & inputs_q
  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  mask_ids = tf.constant(FLAGS.mask_id, shape=[seq_len], dtype=inputs_k.dtype)
  inputs_q = tf.where(can_attend_self, inputs_k, mask_ids)

  if FLAGS.rand_ratio > 0:
    rand_ids = tf.random.uniform([seq_len], maxval=FLAGS.vocab_size,
                                 dtype=inputs_k.dtype)
    change_to_rand = tf.logical_and(
        FLAGS.leak_ratio < random_p,
        random_p < FLAGS.leak_ratio + FLAGS.rand_ratio)
    change_to_rand = tf.logical_and(change_to_rand, is_masked)
    inputs_q = tf.where(change_to_rand, rand_ids, inputs_q)

  return perm_mask_k, perm_mask_q, inputs_k, inputs_q, masked_tokens


def create_target_mapping(example, is_masked, seq_len, num_predict,
                          **kwargs):
  """Create target mapping and retrieve the corresponding targets."""
  indices = tf.range(seq_len, dtype=tf.int64)
  indices = tf.boolean_mask(indices, is_masked)

  ##### extra padding if needed
  actual_num_predict = tf.shape(indices)[0]
  pad_len = num_predict - actual_num_predict

  ##### target_mapping
  target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
  paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
  target_mapping = tf.concat([target_mapping, paddings], axis=0)
  example["target_mapping"] = tf.reshape(target_mapping,
                                         [num_predict, seq_len])

  ##### target mask
  target_mask = tf.concat(
      [tf.ones([actual_num_predict], dtype=tf.float32),
       tf.zeros([pad_len], dtype=tf.float32)],
      axis=0)
  example["target_mask"] = tf.reshape(target_mask, [num_predict])

  ##### Handle fields in kwargs
  for k, v in kwargs.items():
    pad_shape = [pad_len] + v.shape.as_list()[1:]
    tgt_shape = [num_predict] + v.shape.as_list()[1:]
    example[k] = tf.concat([
        tf.boolean_mask(v, is_masked),
        tf.zeros(shape=pad_shape, dtype=v.dtype)], 0)
    example[k].set_shape(tgt_shape)


def chunk_to_sequence(dataset, seq_len):
  """Turn a dataset of doc tfrecords into a dataset of chunked seqeuences."""
  # Flatten the original dataset into a continuous stream and then chunk the
  # continuous stream into segments of fixed length `seq_len`
  if FLAGS.add_double_eos:
    token_len = seq_len - 2
    dataset = dataset.unbatch().repeat().batch(token_len)
    def add_double_eos(example):
      """Add <eos> in the begin and end of the sequence."""
      inputs, type_id = example["inputs"], example["type_id"]
      eos_tensor = tf.constant(FLAGS.eos_id, shape=[1], dtype=inputs.dtype)
      example["inputs"] = tf.concat([eos_tensor, inputs, eos_tensor], 0)
      example["type_id"] = tf.concat([type_id[:1], type_id, type_id[-1:]], 0)

      return example

    dataset = dataset.map(add_double_eos)
  elif FLAGS.add_eos:
    token_len = seq_len - 1
    dataset = dataset.unbatch().repeat().batch(token_len)
    def add_eos(example):
      """Add <eos> at the end of the sequence."""
      inputs, type_id = example["inputs"], example["type_id"]
      eos_tensor = tf.constant(FLAGS.eos_id, shape=[1], dtype=inputs.dtype)
      example["inputs"] = tf.concat([inputs, eos_tensor], 0)
      example["type_id"] = tf.concat([type_id, type_id[-1:]], 0)

      return example

    dataset = dataset.map(add_eos)
  else:
    dataset = dataset.unbatch().repeat().batch(seq_len)

  # Set fixed shape
  def set_shape(example):
    """Give fixed shape to example for TPU."""
    for k in example.keys():
      example[k].set_shape((seq_len))
    return example
  dataset = dataset.map(set_shape)

  return dataset


def create_mlm_target(example, seq_len, num_predict, use_bfloat16):
  """docs."""
  inputs = example["inputs"]

  # sample mask
  is_masked = online_sample_masks(
      inputs, seq_len, num_predict)
  masked_input = discrepancy_correction(inputs, is_masked, seq_len)

  # masked_input, original_input, input_mask
  example["masked_input"] = tf.reshape(masked_input, [seq_len])
  example["inp_mask"] = tf.cast(
      tf.equal(example["masked_input"], FLAGS.mask_id), tf.float32)

  # create target mapping
  create_target_mapping(example, is_masked, seq_len, num_predict,
                        target=inputs)

  # type cast for example
  type_cast(example, use_bfloat16)

  for k, v in example.items():
    tf.logging.info("%s: %s", k, v)

  return example


def mlm_process(dataset, seq_len, num_predict, use_bfloat16):
  """Process input tfrecords into proper format for MLM training."""
  # Get input sequence
  dataset = chunk_to_sequence(dataset, seq_len)

  # Create MLM target
  create_mlm_target_mapper = functools.partial(
      create_mlm_target,
      seq_len=seq_len,
      num_predict=num_predict,
      use_bfloat16=use_bfloat16)
  dataset = dataset.map(create_mlm_target_mapper, num_parallel_calls=64)

  return dataset


def create_xlnet_target(example, seq_len, num_predict, use_bfloat16):
  """docs."""
  inputs = example.pop("inputs")
  type_id = example.pop("type_id")
  is_masked = online_sample_masks(inputs, seq_len, num_predict)

  # permutate the entire sequence together
  perm_mask_k, perm_mask_q, input_k, input_q, is_masked = create_perm(
      inputs, is_masked, seq_len)

  # Set shape
  perm_mask_k.set_shape([seq_len, seq_len])
  perm_mask_q.set_shape([seq_len, seq_len])
  input_k.set_shape([seq_len])
  input_q.set_shape([seq_len])

  # reshape back to fixed shape
  example["input_k"] = input_k
  example["type_id_k"] = type_id
  example["perm_mask_k"] = perm_mask_k

  # create target mapping
  create_target_mapping(example, is_masked, seq_len, num_predict,
                        target=inputs, input_q=input_q, type_id_q=type_id,
                        perm_mask_q=perm_mask_q)

  # type cast for example
  type_cast(example, use_bfloat16)

  for k, v in example.items():
    tf.logging.info("%s: %s", k, v)

  return example


def xlnet_process(dataset, seq_len, num_predict, use_bfloat16):
  """Process input tfrecords into proper format for MLM training."""
  # Get input sequence
  dataset = chunk_to_sequence(dataset, seq_len)

  # Create XLNet target
  create_xlnet_target_mapper = functools.partial(
      create_xlnet_target,
      seq_len=seq_len,
      num_predict=num_predict,
      use_bfloat16=use_bfloat16)
  dataset = dataset.map(create_xlnet_target_mapper, num_parallel_calls=64)

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


def sent_mlm_dataset(params,
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
  """Get sentence level MLM dataset."""
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
  if FLAGS.loss_type == "xlnet":
    dataset = xlnet_process(dataset, seq_len, num_predict, use_bfloat16)
  else:
    dataset = mlm_process(dataset, seq_len, num_predict, use_bfloat16)

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


def semidoc_mlm_dataset(params,
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
  """Get semi-doc level MLM dataset.

  Notes:
  - Each sequence comes from the same document (except for boundary cases).
    This is different from the standard sent-level MLM dataset.
  - No consecutivity is ensured across batches, which is different from the
    standard doc-level MLM dataset.
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
  if FLAGS.loss_type == "xlnet":
    dataset = xlnet_process(dataset, seq_len, num_predict, use_bfloat16)
  else:
    dataset = mlm_process(dataset, seq_len, num_predict, use_bfloat16)

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


def doc_mlm_dataset(params,
                    file_names,
                    num_hosts,
                    num_core_per_host,
                    seq_len,
                    num_predict,
                    is_training,
                    use_bfloat16=False,
                    num_threads=64,
                    record_shuffle_size=256):
  """Get document level MLM dataset."""

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
  if FLAGS.loss_type == "xlnet":
    process_func = xlnet_process
  else:
    process_func = mlm_process

  process_shard = functools.partial(process_func,
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
  func_list = [doc_mlm_dataset, semidoc_mlm_dataset, sent_mlm_dataset]

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

