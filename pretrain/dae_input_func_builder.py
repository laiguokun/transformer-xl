"""Create dae input function for TPUEstimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

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

flags.DEFINE_float("del_ratio", default=0.1,
                   help="#delete / #tok ratio.")
flags.DEFINE_float("ins_ratio", default=0.1,
                   help="#insert / #tok ratio.")
flags.DEFINE_float("rep_ratio", default=0.1,
                   help="#replace / #tok ratio.")
flags.DEFINE_integer("enc_len", default=256,
                     help="Maximum encoder input length.")
flags.DEFINE_integer("dec_len", default=256,
                     help="Maximum decoder input length.")
flags.DEFINE_integer("del_label", default=1,
                     help="Edit label id for delete.")
flags.DEFINE_integer("ins_label", default=2,
                     help="Edit label id for insert.")
flags.DEFINE_integer("rep_label", default=3,
                     help="Edit label id for replace.")


def get_type_id(tgt_len, tgt_idx, type_val):
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

  type_id = example.pop("type_id")
  type_id = tf.concat([type_id, type_id[-1:]], 0)

  ##### Sample positions for different operations
  # (1) Sample deletion positions
  del_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  del_mask = tf.logical_and(del_rand < FLAGS.del_ratio, non_eos_mask)
  non_del_mask = tf.logical_not(del_mask)

  # (2) Sample insertion positions given deletion
  ins_num = int(math.ceil(seq_len * FLAGS.ins_ratio))
  right_shift_del_mask = tf.concat(
      [tf.constant(False, shape=[1], dtype=tf.bool), del_mask[:-1]], axis=0)
  non_ins_mask = tf.logical_or(del_mask, right_shift_del_mask)
  non_ins_mask = tf.logical_or(non_ins_mask, eos_mask)
  ins_uniform = tf.random.uniform(shape=[ins_num, seq_len], minval=0, maxval=1)
  ins_uniform -= 10 * tf.cast(non_ins_mask, tf.float32)
  ins_idx = tf.argmax(ins_uniform, axis=1)
  ins_cnt = tf.reduce_sum(tf.one_hot(ins_idx, seq_len, dtype=tf_int), 0)

  # (3) Sample replace positions given deletion & insertion
  rep_num = int(math.ceil(seq_len * FLAGS.rep_ratio))
  non_rep_mask = tf.logical_or(tf.greater(ins_cnt, 0), non_ins_mask)
  non_rep_mask = tf.logical_or(non_rep_mask, eos_mask)
  rep_uniform = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  rep_uniform -= 10 * tf.cast(non_rep_mask, tf.float32)
  _, rep_idx = tf.math.top_k(rep_uniform, k=rep_num)
  rep_mask = tf.logical_and(
      tf.reduce_sum(tf.one_hot(rep_idx, seq_len, dtype=tf_int), 0) > 0,
      tf.logical_not(non_rep_mask))

  # change replaced locations to <mask>
  rep_input = tf.where(
      rep_mask,
      tf.constant(FLAGS.mask_id, shape=[seq_len], dtype=tf_int),
      inputs)

  ######### construct features
  ori_idx = tf.range(seq_len, dtype=tf_int)

  ##### encoder features
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

  ##### generator features
  is_masked = tf.equal(enc_seq, FLAGS.mask_id)

  # gen_mask_map: extract `num_mask` sub-seq from `enc_len` full-seq
  indices = tf.range(FLAGS.enc_len, dtype=tf_int)
  indices = tf.boolean_mask(indices, is_masked)
  num_mask = rep_num + ins_num
  actual_num_mask = tf.shape(indices)[0]
  map_pad_len = num_mask - actual_num_mask
  gen_mask_map = tf.one_hot(indices, FLAGS.enc_len, dtype=tf.float32)
  map_padding = tf.zeros([map_pad_len, FLAGS.enc_len], dtype=gen_mask_map.dtype)
  gen_mask_map = tf.concat([gen_mask_map, map_padding], axis=0)
  gen_mask_map = tf.reshape(gen_mask_map, [num_mask, FLAGS.enc_len])

  # gen_tgt_mask: only `rep_num` 1s in the sequence of length `num_mask`
  no_ins_inp = tf.scatter_nd(shape=[FLAGS.enc_len],
                             indices=enc_idx[:, None],
                             updates=enc_val)
  is_rep = tf.equal(no_ins_inp, FLAGS.mask_id)
  gen_tgt_mask = tf.boolean_mask(is_rep, is_masked)
  gen_tgt_mask.set_shape([num_mask])

  # gen_tgt: scatter `rep_num` replaced ids to the correct positions in the
  # sequence of total length `num_mask` (others correspond to insertions)
  enc_valid_rep_mask = tf.logical_and(rep_mask, enc_shift_idx < FLAGS.enc_len)
  gen_tgt_val = tf.boolean_mask(inputs, enc_valid_rep_mask)
  gen_tgt_idx = tf.boolean_mask(tf.range(num_mask, dtype=tf_int), gen_tgt_mask)
  gen_tgt = tf.tensor_scatter_nd_update(
      tf.constant(FLAGS.pad_id, shape=[num_mask], dtype=tf_int),
      indices=gen_tgt_idx[:, None],
      updates=gen_tgt_val)

  ##### decoder features
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
      updates=tf.boolean_mask(rep_mask, dec_valid_mask)
  )
  dec_del_mask = tf.scatter_nd(
      shape=[FLAGS.dec_len],
      indices=dec_idx[:, None],
      updates=tf.boolean_mask(del_mask, dec_valid_mask)
  )
  edit_label = tf.cast(dec_ins_mask, tf_int) * FLAGS.ins_label
  edit_label += tf.cast(dec_rep_mask, tf_int) * FLAGS.rep_label
  edit_label += tf.cast(dec_del_mask, tf_int) * FLAGS.del_label

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

