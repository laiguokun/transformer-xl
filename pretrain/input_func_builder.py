"""Create input function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain.data_utils import type_cast
  from google3.experimental.users.zihangd.pretrain.data_utils import format_filename
except ImportError as e:
  from data_utils import type_cast
  from data_utils import format_filename
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS
flags.DEFINE_enum("sample_strategy", default="single_token",
                  enum_values=["single_token", "token_span"],
                  help="Stragey used to sample prediction targets.")

flags.DEFINE_integer("max_tok", default=5,
                     help="Maximum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")
flags.DEFINE_integer("min_tok", default=1,
                     help="Minimum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")


def parse_files_to_dataset(parser, file_names, split, num_hosts,
                           host_id, num_core_per_host, bsz_per_core):
  """Parse a list of file names into a single tf.dataset."""
  if len(file_names) >= num_hosts:
    file_paths = file_names[host_id::num_hosts]
  else:
    file_paths = file_names
  tf.logging.info("Host %d handles %d files", host_id, len(file_paths))

  assert split == "train"
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # file-level shuffle
  if len(file_paths) > 1:
    tf.logging.info("Perform file-level shuffle with size %d", len(file_paths))
    dataset = dataset.shuffle(len(file_paths))

  # `cycle_length` is the number of parallel files that get read.
  cycle_length = min(8, len(file_paths))
  tf.logging.info("Interleave %d files", cycle_length)

  # `sloppy` mode means that the interleaving is not exact. This adds
  # even more randomness to the training pipeline.
  dataset = dataset.apply(
      tf.contrib.data.parallel_interleave(
          tf.data.TFRecordDataset,
          sloppy=True,
          cycle_length=cycle_length))

  buffer_size = 2048
  tf.logging.info("Perform sample-level shuffle with size %d", buffer_size)
  dataset = dataset.shuffle(buffer_size=buffer_size)

  # Note: since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  dataset = dataset.cache().map(parser).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset


def _idx_pair_to_mask(beg_indices, end_indices, inputs, tgt_len, num_predict):
  """Turn beg and end indices into actual mask."""
  non_func_mask = tf.logical_and(tf.not_equal(inputs, FLAGS.eos_id))
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

  return is_masked, target_mask


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


def _single_token_mask(inputs, tgt_len, num_predict):
  """Sample individual tokens as prediction targets."""
  all_indices = tf.range(tgt_len, dtype=tf.int64)
  non_func_mask = tf.not_equal(inputs, FLAGS.eos_id)
  non_func_indices = tf.boolean_mask(all_indices, non_func_mask)

  masked_pos = tf.random_shuffle(non_func_indices)
  masked_pos = tf.contrib.framework.sort(masked_pos[:num_predict])
  target_mask = tf.sparse_to_dense(
      sparse_indices=masked_pos,
      output_shape=[tgt_len],
      sparse_values=1.0,
      default_value=0.0)

  is_masked = tf.cast(target_mask, tf.bool)

  return is_masked, target_mask


def _online_sample_masks(inputs, tgt_len, num_predict):
  """Sample target positions to predict."""
  tf.logging.info("Online sample with strategy: `%s`.", FLAGS.sample_strategy)
  if FLAGS.sample_strategy == "single_token":
    return _single_token_mask(inputs, tgt_len, num_predict)
  elif FLAGS.sample_strategy == "token_span":
    return _token_span_mask(inputs, tgt_len, num_predict)
  else:
    raise NotImplementedError


def _filter_func_tokens(inputs, is_masked):
  """Filter out functional tokens."""
  non_func_tokens = tf.logical_not(tf.equal(inputs, FLAGS.eos_id))
  is_masked = tf.logical_and(is_masked, non_func_tokens)

  target_mask = tf.cast(is_masked, tf.float32)

  return is_masked, target_mask


def discrepancy_correction(inputs, is_masked, tgt_len):
  """Construct the masked input."""
  # 80% MASK, 10% original, 10% random words
  random_p = tf.random.uniform([tgt_len], maxval=1.0)
  mask_array = tf.constant(FLAGS.mask_id, dtype=tf.int64, shape=[tgt_len])
  mask_v = tf.where(random_p < 0.2, inputs, mask_array)
  random_words = tf.random.uniform([tgt_len], maxval=FLAGS.vocab_size,
                                   dtype=tf.int64)
  mask_v = tf.where(random_p < 0.1, random_words, mask_v)

  masked_ids = tf.where(is_masked, mask_v, inputs)

  return masked_ids


def create_target_mapping(example, target, target_mask, seq_len, num_predict):
  """Create target mapping and retrieve the corresponding targets."""
  if num_predict is not None:
    indices = tf.range(seq_len, dtype=tf.int64)
    bool_target_mask = tf.cast(target_mask, tf.bool)
    indices = tf.boolean_mask(indices, bool_target_mask)

    ##### extra padding if needed
    actual_num_predict = tf.shape(indices)[0]
    pad_len = num_predict - actual_num_predict

    ##### target_mapping
    target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
    paddings = tf.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
    target_mapping = tf.concat([target_mapping, paddings], axis=0)
    example["target_mapping"] = tf.reshape(target_mapping,
                                           [num_predict, seq_len])

    ##### target
    target = tf.boolean_mask(target, bool_target_mask)
    paddings = tf.zeros([pad_len], dtype=target.dtype)
    target = tf.concat([target, paddings], axis=0)
    example["target"] = tf.reshape(target, [num_predict])

    ##### target mask
    target_mask = tf.concat(
        [tf.ones([actual_num_predict], dtype=tf.float32),
         tf.zeros([pad_len], dtype=tf.float32)],
        axis=0)
    example["target_mask"] = tf.reshape(target_mask, [num_predict])
  else:
    example["target"] = tf.reshape(target, [seq_len])
    example["target_mask"] = tf.reshape(target_mask, [seq_len])


def get_one_stream_dataset(
    params, num_hosts, num_core_per_host, split, file_names, seq_len,
    num_predict, use_bfloat16=False):
  """Get one-stream dataset."""

  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  #### Function used to parse tfrecord
  def parser(record):
    """function used to parse tfrecord."""

    record_spec = {
        "input": tf.FixedLenFeature([seq_len], tf.int64),
        "type_id": tf.FixedLenFeature([seq_len], tf.int64),
    }

    # retrieve serialized example
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    inputs = example.pop("input")

    # sample mask
    is_masked, target_mask = _online_sample_masks(
        inputs, seq_len, num_predict)
    masked_input = discrepancy_correction(inputs, is_masked, seq_len)

    # masked_input, original_input, input_mask
    example["masked_input"] = tf.reshape(masked_input, [seq_len])
    example["inp_mask"] = tf.cast(
        tf.equal(example["masked_input"], FLAGS.mask_id), tf.float32)

    # create target mapping
    create_target_mapping(example, inputs, target_mask, seq_len, num_predict)

    # type cast for example
    type_cast(example, use_bfloat16)

    for k, v in example.items():
      tf.logging.info("%s: %s", k, v)

    return example

  # Get dataset
  dataset = parse_files_to_dataset(
      parser=parser,
      file_names=file_names,
      split=split,
      num_hosts=num_hosts,
      host_id=host_id,
      num_core_per_host=num_core_per_host,
      bsz_per_core=bsz_per_core)

  return dataset


def get_input_fn(
    tfrecord_dir,
    split,
    bsz_per_host,
    seq_len,
    num_hosts=1,
    num_core_per_host=1,
    uncased=False,
    num_passes=None,
    use_bfloat16=False,
    num_predict=None,
    loss_type="mlm"):
  """Create Estimator input function."""

  # Merge all record infos into a single one
  record_glob_base = format_filename(
      prefix="meta.{}.pass-*".format(split),
      suffix="json*",
      seq_len=seq_len,
      uncased=uncased)

  def _get_num_batch(info):
    if "num_batch" in info:
      return info["num_batch"]
    elif "num_example" in info:
      return info["num_example"] / bsz_per_host
    else:
      raise ValueError("Do not have sample info.")

  record_info = {"num_batch": 0, "filenames": []}

  tfrecord_dirs = tfrecord_dir.split(",")
  tf.logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    tf.logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.gfile.Glob(record_glob))
    tf.logging.info("[%d] Num of record info path: %d",
                    idx, len(record_paths))

    cur_record_info = {"num_batch": 0, "filenames": []}

    for record_info_path in record_paths:
      if num_passes is not None:
        record_info_name = os.path.basename(record_info_path)
        fields = record_info_name.split(".")[2].split("-")
        pass_id = int(fields[-1])
        if pass_id >= num_passes:
          tf.logging.info("Skip pass %d: %s", pass_id, record_info_name)
          continue

      with tf.gfile.Open(record_info_path, "r") as fp:
        info = json.load(fp)
        cur_record_info["num_batch"] += int(_get_num_batch(info))
        cur_record_info["filenames"] += info["filenames"]

    # overwrite directory for `cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    tf.logging.info("[Dir %d] Number of chosen batches: %s",
                    idx, cur_record_info["num_batch"])
    tf.logging.info("[Dir %d] Number of chosen files: %s",
                    idx, len(cur_record_info["filenames"]))
    tf.logging.debug(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  tf.logging.info("Total number of batches: %d", record_info["num_batch"])
  tf.logging.info("Total number of files: %d", len(record_info["filenames"]))
  tf.logging.debug(record_info["filenames"])

  kwargs = dict(
      num_hosts=num_hosts,
      num_core_per_host=num_core_per_host,
      split=split,
      file_names=record_info["filenames"],
      seq_len=seq_len,
      use_bfloat16=use_bfloat16,
      num_predict=num_predict)

  if loss_type in ["mlm"]:
    get_dataset = get_one_stream_dataset
  else:
    raise NotImplementedError

  def input_fn(params):
    """Input function wrapper."""
    assert params["batch_size"] * num_core_per_host == bsz_per_host

    dataset = get_dataset(params=params, **kwargs)

    return dataset

  return input_fn, record_info

