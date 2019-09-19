"""Create LM input function for TPUEstimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.lm.data_utils import convert_example
  from google3.experimental.users.zihangd.lm.data_utils import format_filename
except ImportError as e:
  from data_utils import convert_example
  from data_utils import format_filename
# pylint: enable=g-import-not-at-top


def parse_files_to_dataset(parser, file_names, split, num_hosts,
                           host_id, num_core_per_host, bsz_per_core):
  """Parse file names into a tf.Dataset."""
  if len(file_names) >= num_hosts:
    file_paths = file_names[host_id::num_hosts]
  else:
    file_paths = file_names

  tf.logging.info("Host %d handles %d files:", host_id, len(file_paths))

  assert split == "train"
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # file-level shuffle
  if len(file_paths) > 1:
    dataset = dataset.shuffle(len(file_paths))

  # Note: cannot perform sample-level shuffle here because this will violate
  # the consecutive requirement of data stream.
  dataset = tf.data.TFRecordDataset(dataset)

  dataset = dataset.cache().map(parser).repeat()
  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset


def get_lm_dataset(params, num_hosts, num_core_per_host, split, file_names,
                   seq_len, use_bfloat16=False):
  """Get lm dataset."""

  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  #### Function used to parse tfrecord
  def parser(record):
    """function used to parse tfrecord."""

    record_spec = {
        "inputs": tf.FixedLenFeature([seq_len], tf.int64),
        "target": tf.FixedLenFeature([seq_len], tf.int64),
    }

    # retrieve serialized example
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    convert_example(example, use_bfloat16)

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
    lower_case=False,
    num_passes=None,
    use_bfloat16=False):
  """Create Estimator input function."""

  # Merge all record infos into a single one
  record_glob_base = format_filename(
      prefix="meta.{}.pass-*".format(split),
      suffix="json*",
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      lower_case=lower_case)

  def _get_num_batch(info):
    if "num_batch" in info:
      return info["num_batch"]
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
        if num_passes is not None:
          eff_num_passes = min(num_passes, len(info["filenames"]))
          ratio = eff_num_passes / len(info["filenames"])
          cur_record_info["num_batch"] += int(_get_num_batch(info) * ratio)
          cur_record_info["filenames"] += info["filenames"][:eff_num_passes]
        else:
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
    tf.logging.info(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  tf.logging.info("Total number of batches: %d",
                  record_info["num_batch"])
  tf.logging.info("Total number of files: %d",
                  len(record_info["filenames"]))
  tf.logging.info(record_info["filenames"])

  kwargs = dict(
      num_hosts=num_hosts,
      num_core_per_host=num_core_per_host,
      split=split,
      file_names=record_info["filenames"],
      seq_len=seq_len,
      use_bfloat16=use_bfloat16)

  def input_fn(params):
    """docs."""
    assert params["batch_size"] * num_core_per_host == bsz_per_host

    dataset = get_lm_dataset(params=params, **kwargs)

    return dataset

  return input_fn, record_info

