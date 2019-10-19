"""Create tfrecords for LM training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  import google3.experimental.users.zihangd.pretrain.data_utils as data_utils
  import google3.experimental.users.zihangd.pretrain.tokenization as tokenization
  import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

  run_internal = True
except ImportError as e:
  import tensorflow as tf
  import data_utils
  import tokenization

  run_internal = False
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS

flags.DEFINE_integer("type_id", default=0,
                     help="Language type id.")
flags.DEFINE_integer("min_doc_len", default=1,
                     help="Minimum document length.")

flags.DEFINE_bool("add_eos", False,
                  help="whether to append EOS at the end of a line.")
flags.DEFINE_bool("add_double_eos", False,
                  help="whether to append EOS at the begin and end of a line.")

flags.DEFINE_string("input_glob", "data/example/*.txt",
                    help="Input file glob.")
flags.DEFINE_string("save_dir", "proc_data/example",
                    help="Directory for saving the processed data.")
flags.DEFINE_enum("split", "train", ["train", "valid", "test"],
                  help="Save the data as which split.")

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
  data_prefix = "data.{}.doc".format(FLAGS.split)
  data_suffix = "tfrecord-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  tfrecord_name = data_utils.format_filename(
      prefix=data_prefix,
      suffix=data_suffix,
      uncased=FLAGS.uncased,
  )
  tfrecord_path = os.path.join(save_dir, tfrecord_name)

  return tfrecord_path


def _meta_path(save_dir):
  """Get meta path."""
  meta_prefix = "meta.{}.doc".format(FLAGS.split)
  meta_suffix = "json-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  meta_name = data_utils.format_filename(
      prefix=meta_prefix,
      suffix=meta_suffix,
      uncased=FLAGS.uncased,
  )
  meta_path = os.path.join(save_dir, meta_name)

  return meta_path


def _create_data(input_paths, tokenizer):
  """Load data and call corresponding create_func."""

  all_data = []
  total_doc_cnt = 0
  for input_path in input_paths:
    cur_docs = data_utils.read_docs(input_path, tokenizer)

    if cur_docs:
      total_doc_cnt += len(cur_docs)
      all_data.extend(cur_docs)

  tf.logging.info("[Task %d] Total number docs: %d", FLAGS.task, total_doc_cnt)

  create_ordered_lm_tfrecords(
      save_dir=FLAGS.save_dir,
      data=all_data,
  )


def create_data(_):
  """create pretraining data (tfrecords)."""
  tokenizer = tokenization.get_tokenizer()

  # Make workdirs
  if not tf.gfile.Exists(FLAGS.save_dir):
    tf.gfile.MakeDirs(FLAGS.save_dir)

  # Interleavely split the work into FLAGS.num_task splits
  file_paths = sorted(tf.gfile.Glob(FLAGS.input_glob))
  tf.logging.info("Use glob: %s", FLAGS.input_glob)
  tf.logging.info("Find %d files", len(file_paths))

  task_file_paths = file_paths[FLAGS.task::FLAGS.num_task]
  if not task_file_paths:
    tf.logging.info("Exit: task %d has no file to process.", FLAGS.task)
    return

  tf.logging.info("Task %d process %d files:", FLAGS.task, len(task_file_paths))
  for task_file in task_file_paths:
    tf.logging.debug("  - %s", task_file)

  _create_data(task_file_paths, tokenizer)


def create_ordered_lm_tfrecords(save_dir, data):
  """create ordered language modeling tfrecords from numpy array."""
  tf.logging.info("===== Create ordered LM tfrecords =====")

  ##### Create record writer
  tfrecord_path = _tfrecord_path(save_dir)
  record_writer = tf.python_io.TFRecordWriter(tfrecord_path)
  tf.logging.info("Start writing tfrecord %s.", tfrecord_path)

  num_example = 0
  for doc in data:
    if num_example % 10000 == 0:
      tf.logging.info("Processing example %d.", num_example)

    pos_seq = [np.arange(len(sent), dtype=np.int64) for sent in doc]
    pos_seq = np.concatenate(pos_seq)

    inputs = np.concatenate(doc)

    feature = {
        "inputs": _int64_feature(inputs),
        "pos_seq": _int64_feature(pos_seq),
        "type_id": _int64_feature([FLAGS.type_id]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    record_writer.write(example.SerializeToString())

    num_example += 1

  record_writer.close()
  tf.logging.info("Done writing %s. Num of batches: %d",
                  tfrecord_path, num_example)

  ##### dump record information
  meta_info = {
      "filenames": [os.path.basename(tfrecord_path)],
      "num_example": num_example
  }
  meta_path = _meta_path(save_dir)
  with tf.io.gfile.GFile(meta_path, "w") as fp:
    json.dump(meta_info, fp)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(create_data)
