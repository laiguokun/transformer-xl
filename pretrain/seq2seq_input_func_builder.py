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
  from google3.experimental.users.zihangd.pretrain.data_utils import format_filename
  from google3.experimental.users.zihangd.pretrain.generator_utils import pack_dataset
except ImportError as e:
  from data_utils import type_cast
  from data_utils import format_filename
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


def pad_for_tpu(shapes_dict, max_length, filler_len_dict=None):
  """Pads unknown features' dimensions for TPU."""
  padded_shapes = {}

  def pad_one_shape(shape, none_filler):
    return [
        (dim if dim is not None else none_filler) for dim in shape.as_list()
    ]

  for key, shape in six.iteritems(shapes_dict):
    if filler_len_dict is not None and key in filler_len_dict:
      filler_length = min(filler_len_dict[key], max_length)
    else:
      filler_length = max_length

    padded_shapes[key] = pad_one_shape(shape, filler_length)

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


def reorder_dataset(dataset, max_length):
  def reorder(example):
    # using this for debug
    # ret = example
    ret = {}
    src = example["source"]
    tgt = example["target"]
    src_seg = example["source_segmentation"]
    tgt_seg = example["target_segmentation"]
    src_pos = example["source_position"]
    tgt_pos = example["target_position"]

    # segment_num = max_length // 2
    segment_num = tf.reduce_max(src_seg) + 1
    src_seg_cnt = tf.reduce_sum(
        tf.one_hot(src_seg, segment_num, dtype=tf.int32), axis=0)
    tgt_seg_cnt = tf.reduce_sum(
        tf.one_hot(tgt_seg, segment_num, dtype=tf.int32), axis=0)
    max_seg_cnt = tf.math.maximum(src_seg_cnt, tgt_seg_cnt)

    # remove cnt of 0 seg
    max_seg_cnt = tf.tensor_scatter_nd_update(
        max_seg_cnt,
        indices=tf.constant([0], tf.int32)[:, None],
        updates=tf.constant([0], max_seg_cnt.dtype)
    )
    max_seg_cnt = max_seg_cnt * 2
    seg_start_idx = tf.cumsum(max_seg_cnt, exclusive=True)
    seg_end_idx = tf.cumsum(max_seg_cnt)
    is_valid = tf.cumsum(max_seg_cnt) <= max_length
    max_valid_seg = tf.reduce_sum(tf.cast(is_valid, tf.int32))
    valid_src_mask = tf.logical_and(src_seg > 0, src_seg < max_valid_seg)
    valid_tgt_mask = tf.logical_and(tgt_seg > 0, tgt_seg < max_valid_seg)

    # even pos is source, odd pos is target
    src_valid = tf.boolean_mask(src, valid_src_mask)
    src_valid_seg = tf.boolean_mask(src_seg, valid_src_mask)
    src_valid_pos = tf.boolean_mask(src_pos * 2, valid_src_mask)
    tgt_valid = tf.boolean_mask(tgt, valid_tgt_mask)
    tgt_valid_seg = tf.boolean_mask(tgt_seg, valid_tgt_mask)
    tgt_valid_pos = tf.boolean_mask(tgt_pos * 2 + 1, valid_tgt_mask)

    src_map = tf.one_hot(src_valid_seg, segment_num, dtype=tf.int32)
    src_seg_shift = tf.einsum("s,ls->l", seg_start_idx, src_map)
    src_valid_pos += src_seg_shift
    tgt_map = tf.one_hot(tgt_valid_seg, segment_num, dtype=tf.int32)
    tgt_seg_shift = tf.einsum("s,ls->l", seg_start_idx, tgt_map)
    tgt_valid_pos += tgt_seg_shift

    # get inputs
    inputs = tf.scatter_nd(
        shape=[max_length],
        indices=src_valid_pos[:, None],
        updates=src_valid)
    inputs = tf.tensor_scatter_nd_update(
        inputs,
        indices=tgt_valid_pos[:, None],
        updates=tgt_valid)

    # get targets
    targets = tgt * tf.cast(valid_tgt_mask, dtype=tgt.dtype)
    targets = tf.concat([targets[1:], targets[:1]], axis=0)
    non_pad_mask = tf.not_equal(targets, 0)
    all_eos = tf.constant(FLAGS.eos_id, shape=targets.shape,
                          dtype=targets.dtype)
    targets = tf.where(non_pad_mask, targets, all_eos)

    loss_mask = tf.cast(non_pad_mask, tf.float32)

    # get segmentation
    seg_start_idx_valid = seg_start_idx[1:max_valid_seg]
    seg_end_idx_valid = seg_end_idx[1:max_valid_seg]
    seg = tf.scatter_nd(
        shape=[max_length],
        indices=seg_start_idx_valid[:, None],
        updates=tf.ones(shape=[max_valid_seg - 1]))
    seg = tf.cumsum(seg)
    pad_len = max_length - seg_end_idx_valid[-1]
    seg = tf.tensor_scatter_nd_update(
        seg,
        indices=tf.range(seg_end_idx_valid[-1], max_length)[:, None],
        updates=tf.zeros(shape=[pad_len])
    )
    seg = tf.cast(seg, tf.int32)

    seg_map = tf.one_hot(seg, segment_num, dtype=tf.int32)
    seg_pos_shift = tf.einsum("s,ls->l", seg_start_idx, seg_map)

    # get position
    pos = tf.cumsum(tf.ones(shape=[max_length], dtype=tf.int32), exclusive=True)
    pos = pos - seg_pos_shift
    pos = tf.tensor_scatter_nd_update(
        pos,
        indices=tf.range(seg_end_idx_valid[-1], max_length)[:, None],
        updates=tf.zeros(shape=[pad_len], dtype=pos.dtype)
    )
    pos = tf.cast(pos, tf.int32)
    ret["inputs"] = inputs
    ret["targets"] = targets
    ret["loss_mask"] = loss_mask
    ret["segmentation"] = seg
    ret["position"] = pos

    return ret

  return dataset.map(reorder)


def get_dataset(params,
                num_hosts,
                data_files,
                is_training,
                max_length,
                min_length=1,
                use_bfloat16=False,
                preprocess=True,
                num_threads=64,
                prefetch_size=None,
                record_shuffle_size=10240,
                batch_shuffle_size=2048,
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
  ## Shuffle files only for training examples.
  if is_training:
    tf.logging.info("File level shuffle with size: %d", len(data_files))
    dataset = dataset.shuffle(len(data_files))

  # Create data-set from files by parsing, pre-processing and interleaving.
  if is_training:
    cycle_length = min(64, len(data_files))
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _load_records_and_preprocess,
            sloppy=True,
            cycle_length=cycle_length))
  else:
    dataset = _load_records_and_preprocess(dataset)

  dataset = dataset.take(max_records)

  ## Shuffle records only for training examples.
  if is_training:
    tf.logging.info("Record level shuffle with size: %d", record_shuffle_size)
    dataset = dataset.shuffle(record_shuffle_size)

  ##### IMPORTANT #####
  if FLAGS.pack_dataset:
    if FLAGS.seq2seq_type == "dec_only" and FLAGS.rel_attn:
      def concat_src_tgt(example):
        src = example.pop("source")
        tgt = example.pop("target")
        example["inputs"] = tf.concat([src, tgt], 0)
        example["type_ids"] = tf.concat([
            tf.zeros_like(src, dtype=src.dtype),
            tf.ones_like(tgt, dtype=tgt.dtype)], 0)
        return example
      dataset = dataset.map(concat_src_tgt,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = pack_dataset(
          dataset, max_length, keys=["inputs", "type_ids"],
          use_custom_ops=FLAGS.use_custom_ops)
      def remove_auxiliary_structure(example):
        example.pop("type_ids_segmentation")
        example.pop("type_ids_position")
        return example
      def reset_pos(example):
        """reset the pos in one segment."""
        # input pos: [0, 1, 2, 3, 4, 5, 0, ....]
        # input type_ids: [0, 0, 0, 1, 1, 1, 0, ....]
        # target pos:
        pos = example["inputs_position"]
        type_ids = example["type_ids"]
        seg = example["inputs_segmentation"]
        seg_num = tf.reduce_max(seg) + 1
        source_mask = tf.cast(1 - type_ids, pos.dtype)
        seg_map = tf.one_hot(seg, seg_num, dtype=source_mask.dtype)
        source_cnt = tf.einsum("l,ls->s", source_mask, seg_map)
        source_cnt = tf.einsum("s,ls->l", source_cnt, seg_map)
        target_pos = pos - source_cnt
        new_pos = tf.where(tf.cast(source_mask, tf.bool), pos, target_pos)
        example["inputs_position"] = new_pos
        return example

      dataset = dataset.map(reset_pos,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

      def get_target(example):
        """Deal with target."""
        seq_len = tf.shape(example["inputs"])[0]
        targets_mask = tf.cast(example["type_ids"], tf.bool)
        targets = tf.boolean_mask(example["inputs"], targets_mask)

        pad_len = (seq_len -
                   tf.cast(tf.reduce_sum(example["type_ids"]),
                           dtype=seq_len.dtype))
        targets = tf.concat(
            [targets, tf.zeros(shape=[pad_len], dtype=targets.dtype)],
            axis=0)
        targets_idx = tf.boolean_mask(tf.range(seq_len), targets_mask)
        targets_idx = tf.concat(
            [targets_idx, tf.zeros(shape=[pad_len], dtype=targets_idx.dtype)],
            axis=0)

        targets_map = tf.one_hot(targets_idx, seq_len)

        # padding
        non_pad_mask = tf.not_equal(targets, 0)
        all_eos = tf.ones(shape=[seq_len], dtype=targets.dtype) * FLAGS.eos_id
        # Replace all <pad> (/P) with <eos> (/S)
        #   - target : /S a1 a2 a3 /S b1 b2 /S c1 c2 /P /P
        #   - tmptgt : /S a1 a2 a3 /S b1 b2 /S c1 c2 /S /S
        tmptgt = tf.where(non_pad_mask, targets, all_eos)
        # Shift the `tmptgt` to form the (next-step) prediction target
        #   - target   : \S a1 a2 a3 \S b1 b2 \S c1 c2 \P \P
        #   - pred_tgt : a1 a2 a3 \S b1 b2 \S c1 c2 \S \S \S
        pred_tgt = tf.concat([tmptgt[1:], tmptgt[:1]], axis=0)
        example["targets"] = pred_tgt
        example["targets_map"] = targets_map
        loss_mask = tf.cast(non_pad_mask, tf.float32)
        example["loss_mask"] = loss_mask
        return example

      dataset = dataset.map(get_target,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.map(remove_auxiliary_structure,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      dataset = pack_dataset(
          dataset, max_length, keys=["source", "target"],
          use_custom_ops=FLAGS.use_custom_ops)
  ##### IMPORTANT #####

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  dataset = dataset.repeat()

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
    tf.logging.info("Batch level shuffle with size: %d", batch_shuffle_size)
    dataset = dataset.shuffle(batch_shuffle_size)

  ##### Cast to float16 if needed
  type_cast_ = functools.partial(type_cast, use_bfloat16=use_bfloat16)
  dataset = dataset.map(type_cast_,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return dataset


def get_input_fn(
    tfrecord_dir,
    split,
    max_length,
    num_hosts=1,
    uncased=False,
    use_bfloat16=False,
    **kwargs):
  """Create Estimator input function."""

  # Merge all record infos into a single one
  record_glob_base = format_filename(
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
      with tf.io.gfile.GFile(record_info_path, "r") as fp:
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
      use_bfloat16=use_bfloat16,
      **kwargs)

  def input_fn(params):
    """Input function wrapper."""
    dataset = get_dataset(params=params, **kwargs)

    return dataset

  return input_fn, record_info

