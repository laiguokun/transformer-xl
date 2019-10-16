"""Create input function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import six
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain.data_utils import type_cast
  from google3.experimental.users.zihangd.pretrain.data_utils import seq2seq_filename
  from google3.experimental.users.zihangd.pretrain.generator_utils import pack_dataset
except ImportError as e:
  from data_utils import type_cast
  from data_utils import seq2seq_filename
  from generator_utils import pack_dataset
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS
flags.DEFINE_integer("max_src_len", default=128,
                     help="Maximum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")
flags.DEFINE_integer("max_tgt_len", default=128,
                     help="Maximum number of tokens to sample in a span."
                     "Effective when token_span strategy is used.")
flags.DEFINE_bool("pack_dataset", default=True,
                  help="")
flags.DEFINE_bool("use_custom_ops", default=False,
                  help="")


def example_length(example):
  length = 0
  # Length of the example is the maximum length of the feature lengths
  for _, v in sorted(six.iteritems(example)):
    # For images the sequence length is the size of the spatial dimensions.
    feature_length = tf.shape(v)[0]
    if len(v.get_shape()) > 2:
      feature_length = tf.shape(v)[0] * tf.shape(v)[1]
    length = tf.maximum(length, feature_length)
  return length


def example_valid_size(example, min_length, max_length):
  length = example_length(example)
  return tf.logical_and(
      length >= min_length,
      length <= max_length,
  )


def decode_example(serialized_example):
  """Return a dict of Tensors from a serialized tensorflow.Example."""
  data_fields = {
      "source": tf.VarLenFeature(tf.int64),
      "target": tf.VarLenFeature(tf.int64)
  }

  data_items_to_decoders = {
      field: tf.contrib.slim.tfexample_decoder.Tensor(field)
      for field in data_fields
  }

  decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
      data_fields, data_items_to_decoders)

  decode_items = list(sorted(data_items_to_decoders))
  decoded = decoder.decode(serialized_example, items=decode_items)
  return dict(zip(decode_items, decoded))


def pad_for_tpu(shapes_dict, max_length):
  """Pads unknown features' dimensions for TPU."""
  padded_shapes = {}

  def get_filler(specified_max_length):
    if not specified_max_length:
      return max_length
    return min(specified_max_length, max_length)

  source_none_filler = get_filler(FLAGS.max_src_len)
  target_none_filler = get_filler(FLAGS.max_tgt_len)

  def pad_one_shape(shape, none_filler):
    return [
        (dim if dim is not None else none_filler) for dim in shape.as_list()
    ]

  for key, shape in six.iteritems(shapes_dict):
    if key == "source":
      padded_shapes[key] = pad_one_shape(shape, source_none_filler)
    elif key == "target":
      padded_shapes[key] = pad_one_shape(shape, target_none_filler)
    else:
      padded_shapes[key] = pad_one_shape(shape, max_length)

  return padded_shapes


def run_preprocess(dataset, interleave=True):
  """Runtime preprocessing on the whole dataset.

  Return a tf.data.Dataset -- the preprocessed version of the given one.
  By default this function calls preprocess_example.
  Args:
    dataset: the Dataset of already decoded but not yet preprocessed features.
    interleave: bool, whether to use parallel_interleave, which is faster
      but will alter the order of samples non-deterministically, or flat_map,
      which is slower but will preserve the sample order.
  Returns:
    a Dataset
  """
  def _preprocess(example):
    if "source" in example and FLAGS.max_src_len > 0:
      example["source"] = example["source"][:FLAGS.max_src_len]
    if "target" in example and FLAGS.max_tgt_len > 0:
      example["target"] = example["target"][:FLAGS.max_tgt_len]
    if not isinstance(example, tf.data.Dataset):
      example = tf.data.Dataset.from_tensors(example)
    return example

  if interleave:
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _preprocess, sloppy=True, cycle_length=8))
  else:
    dataset = dataset.flat_map(_preprocess)

  return dataset


def get_dataset(params,
                num_hosts,
                data_files,
                is_training,
                max_length,
                min_length=1,
                preprocess=True,
                num_threads=64,
                prefetch_size=None,
                record_shuffle_size=2048,
                batch_shuffle_size=512,
                use_bfloat16=False,
                max_records=-1):
  """Build a Dataset for this problem."""
  #### Split data files across hosts
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0
  if len(data_files) >= num_hosts:
    data_files = data_files[host_id::num_hosts]
  tf.logging.info("Host %d handles %d files", host_id, len(data_files))

  # Functions used in dataset transforms below. `filenames` can be either a
  # `tf.string` tensor or `tf.data.Dataset` containing one or more filenames.
  def _load_records_and_preprocess(filenames):
    """Reads files from a string tensor or a dataset of filenames."""
    # Load records from file(s) with an 8MiB read buffer.
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
    # Decode.
    dataset = dataset.map(decode_example, num_parallel_calls=num_threads)
    # Preprocess if requested.
    # Note that preprocessing should happen per-file as order may matter.
    if preprocess:
      dataset = run_preprocess(dataset, interleave=is_training)
    return dataset

  dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
  # Create data-set from files by parsing, pre-processing and interleaving.
  if is_training:
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _load_records_and_preprocess, sloppy=True, cycle_length=8))
  else:
    dataset = _load_records_and_preprocess(dataset)

  dataset = dataset.take(max_records)

  ## Shuffle records only for training examples.
  if is_training:
    dataset = dataset.shuffle(record_shuffle_size)

  ##### IMPORTANT #####
  if FLAGS.pack_dataset:
    dataset = pack_dataset(
        dataset, max_length, keys=["source", "target"],
        use_custom_ops=FLAGS.use_custom_ops)
  ##### IMPORTANT #####

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  dataset = dataset.repeat()

  type_cast_ = functools.partial(type_cast, use_bfloat16=use_bfloat16)
  dataset = dataset.map(type_cast_, num_parallel_calls=num_threads)

  ##### Filter out invalid (in terms of length) records
  tpu_valid_size = functools.partial(example_valid_size,
                                     min_length=min_length,
                                     max_length=max_length)
  dataset = dataset.filter(tpu_valid_size)

  ##### Batching
  padded_shapes = pad_for_tpu(dataset.output_shapes, max_length)
  # on TPU, we use params["batch_size"], which specifies the number of
  # examples across all datashards
  batch_size = params["batch_size"]
  dataset = dataset.padded_batch(
      batch_size, padded_shapes, drop_remainder=True)

  if is_training:
    dataset = dataset.shuffle(batch_shuffle_size)

  return dataset


def get_input_fn(
    tfrecord_dir,
    split,
    max_length,
    num_hosts=1,
    uncased=False,
    num_threads=64,
    use_bfloat16=False):
  """Create Estimator input function."""

  # Merge all record infos into a single one
  record_glob_base = seq2seq_filename(
      prefix="meta.{}".format(split),
      suffix="json*",
      uncased=uncased)

  record_info = {"num_example": 0, "filenames": []}

  tfrecord_dirs = tfrecord_dir.split(",")
  tf.logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    tf.logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.gfile.Glob(record_glob))
    tf.logging.info("[%d] Num of record info path: %d",
                    idx, len(record_paths))

    cur_record_info = {"num_example": 0, "filenames": []}

    for record_info_path in record_paths:
      with tf.gfile.Open(record_info_path, "r") as fp:
        info = json.load(fp)
        cur_record_info["num_example"] += info["num_example"]
        cur_record_info["filenames"] += info["filenames"]

    # overwrite directory for `cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    tf.logging.info("[Dir %d] Number of chosen batches: %s",
                    idx, cur_record_info["num_example"])
    tf.logging.info("[Dir %d] Number of chosen files: %s",
                    idx, len(cur_record_info["filenames"]))
    tf.logging.debug(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_example"] += cur_record_info["num_example"]
    record_info["filenames"] += cur_record_info["filenames"]

  tf.logging.info("Total number of batches: %d", record_info["num_example"])
  tf.logging.info("Total number of files: %d", len(record_info["filenames"]))
  tf.logging.debug(record_info["filenames"])

  kwargs = dict(
      data_files=record_info["filenames"],
      num_hosts=num_hosts,
      is_training=split == "train",
      max_length=max_length,
      num_threads=num_threads,
      use_bfloat16=use_bfloat16)

  def input_fn(params):
    """Input function wrapper."""
    dataset = get_dataset(params=params, **kwargs)

    return dataset

  return input_fn, record_info

