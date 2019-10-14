"""Create tfrecord for pretraining."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  from google3.experimental.users.zihangd.pretrain.data_utils import format_filename
  from google3.experimental.users.zihangd.pretrain.tokenization import get_tokenizer
  import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import
except ImportError as e:
  import tensorflow as tf
  from data_utils import format_filename
  from tokenization import get_tokenizer
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS
flags.DEFINE_integer("min_doc_len", 1,
                     help="Minimum document length allowed.")
flags.DEFINE_integer("seq_len", 512,
                     help="Sequence length.")

flags.DEFINE_string("input_glob", "data/example/*.txt",
                    help="Input file glob.")
flags.DEFINE_string("save_dir", "proc_data/example",
                    help="Directory for saving the processed data.")
flags.DEFINE_enum("split", "train", ["train", "dev", "test"],
                  help="Save the data as which split.")
flags.DEFINE_integer("type_id", default=0,
                     help="Language type id.")

flags.DEFINE_integer("pass_id", 0, help="ID of the current pass."
                     "Different passes sample different negative segment.")
flags.DEFINE_integer("num_task", 1, help="Number of total tasks.")
flags.DEFINE_integer("task", 0, help="The Task ID. This value is used when "
                     "using multiple workers to identify each worker.")

flags.DEFINE_string("master", "local",
                    help="BNS name of the TensorFlow master to use.")
flags.DEFINE_integer("ps_tasks", 0,
                     help="Number of tasks in the parameter server job.")


def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _tfrecord_path(save_dir):
  """Get tfrecord path."""
  data_prefix = "data.{}.pass-{}".format(FLAGS.split, FLAGS.pass_id)
  data_suffix = "tfrecord-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  tfrecord_name = format_filename(
      prefix=data_prefix,
      suffix=data_suffix,
      seq_len=FLAGS.seq_len,
      uncased=FLAGS.uncased,
  )
  tfrecord_path = os.path.join(save_dir, tfrecord_name)

  return tfrecord_path


def _meta_path(save_dir):
  """Get meta path."""
  meta_prefix = "meta.{}.pass-{}".format(FLAGS.split, FLAGS.pass_id)
  meta_suffix = "json-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  meta_name = format_filename(
      prefix=meta_prefix,
      suffix=meta_suffix,
      seq_len=FLAGS.seq_len,
      uncased=FLAGS.uncased,
  )
  meta_path = os.path.join(save_dir, meta_name)

  return meta_path


def _create_data(input_paths, tokenizer):
  """Load data and call corresponding create_func."""
  # Assume documents are separated by blank lines and each input shard
  # corresponds to a document.
  input_shards = []

  # monitor doc length and number of tokens
  doc_length = []
  total_num_tok = 0

  for input_path in input_paths:
    # refresh for each file
    input_data = []
    end_of_doc = False
    line_cnt = 0

    tf.logging.info("Start processing %s", input_path)
    for line in tf.io.gfile.GFile(input_path, "r"):
      if line_cnt % 100000 == 0:
        tf.logging.info("Loading line %d", line_cnt)

      if not line.strip():
        # encounter an empty line (end of a document)
        end_of_doc = True

        cur_sent = []
      else:
        cur_sent = tokenizer.convert_text_to_ids(line.strip())

      if cur_sent:
        input_data.extend(cur_sent)

      if end_of_doc:
        # monitor over doc lengths
        doc_length.append(len(input_data))

        # only retain docs longer than `min_doc_len`
        if len(input_data) >= max(FLAGS.min_doc_len, 1):
          input_array = np.array(input_data, dtype=np.int64)
          input_shards.append(input_array)
          total_num_tok += len(input_data)

        # refresh working structs
        input_data = []
        end_of_doc = False

      line_cnt += 1

    #### Deal with the leafover if any
    if input_data:
      doc_length.append(len(input_data))
      if len(input_data) >= max(FLAGS.min_doc_len, 1):
        input_array = np.array(input_data, dtype=np.int64)
        input_shards.append(input_array)
        total_num_tok += len(input_data)

    tf.logging.info("Finish %s with %d lines.", input_path, line_cnt)

  tf.logging.info("[Task %d] Total number tokens: %d", FLAGS.task,
                  total_num_tok)

  hist, bins = np.histogram(doc_length,
                            bins=[0, 64, 128, 256, 512, 1024, 2048, 2 << 30])
  percent = hist / np.sum(hist)
  tf.logging.info("Doc length histogram:")
  for pct, l, r in zip(percent, bins[:-1], bins[1:]):
    tf.logging.info("  - [%d, %d]: %.4f", l, r, pct)

  # Randomly shuffle input shards (with a fixed but unique random seed)
  np.random.seed(100 * FLAGS.task + FLAGS.pass_id)

  perm_indices = np.random.permutation(len(input_shards))
  tf.logging.debug("Using perm indices %s for pass %d",
                   perm_indices.tolist(), FLAGS.pass_id)

  # concat into a flat np.ndarray
  input_data = np.concatenate([input_shards[idx] for idx in perm_indices])
  create_seq2seq_tfrecords(
      save_dir=FLAGS.save_dir,
      data=input_data,
      tokenizer=tokenizer,
  )


def create_seq2seq_tfrecords(save_dir, data, tokenizer):
  """create seq2seq tfrecords from numpy array."""
  # Notes:
  # - Each sequence is formatted as [<eos> A <eos>] sharing the same type_id

  ##### Prepare data
  tf.logging.info("Raw data shape %s.", data.shape)

  ##### Create record writer
  tfrecord_path = _tfrecord_path(save_dir)
  record_writer = tf.io.TFRecordWriter(tfrecord_path)
  tf.logging.info("Start writing tfrecord to %s.", tfrecord_path)

  ##### Create tfrecord
  data_len = data.shape[0]
  segment_len = FLAGS.seq_len - 2
  eos_array = np.array([tokenizer.eos_id])

  i = 0
  num_example = 0
  while i + segment_len <= data_len:
    if num_example % 10000 == 0:
      tf.logging.info("Processing example %d", num_example)

    ##### create `input`
    cur_seg = data[i: i + segment_len]
    cat_data = np.concatenate([eos_array, cur_seg, eos_array])
    type_id = [FLAGS.type_id] * FLAGS.seq_len

    ##### final check
    assert cat_data.shape[0] == FLAGS.seq_len

    feature = {
        "input": _int64_feature(cat_data),
        "type_id": _int64_feature(type_id),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    record_writer.write(example.SerializeToString())

    # update the number of examples
    num_example += 1

    # update the new begin index
    i += segment_len

  record_writer.close()
  tf.logging.info("Done writing %s. Num of examples: %d",
                  tfrecord_path, num_example)

  ##### dump record information
  meta_info = {
      "filenames": [os.path.basename(tfrecord_path)],
      "num_example": num_example
  }
  meta_path = _meta_path(save_dir)
  with tf.io.gfile.GFile(meta_path, "w") as fp:
    json.dump(meta_info, fp)

  return num_example


def main(_):
  """create pretraining data (tfrecords)."""
  # Load tokenizer
  tokenizer = get_tokenizer()

  # Make workdirs
  if not tf.io.gfile.exists(FLAGS.save_dir):
    tf.io.gfile.makedirs(FLAGS.save_dir)

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")
  if not tf.io.gfile.exists(tfrecord_dir):
    tf.io.gfile.makedirs(tfrecord_dir)

  # Interleavely split the work into FLAGS.num_task splits
  file_paths = sorted(tf.io.gfile.glob(FLAGS.input_glob))
  tf.logging.info("Use glob: %s", FLAGS.input_glob)
  tf.logging.info("Find %d files: %s", len(file_paths), file_paths)

  task_file_paths = file_paths[FLAGS.task::FLAGS.num_task]
  if not task_file_paths:
    tf.logging.info("Exit: task %d has no file to process.", FLAGS.task)
    return

  tf.logging.info("Task %d process %d files: %s",
                  FLAGS.task, len(task_file_paths), task_file_paths)

  _create_data(task_file_paths, tokenizer)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
