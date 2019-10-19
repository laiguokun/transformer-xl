"""Create tfrecord for pretraining."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

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

flags.DEFINE_string("input_glob", "data/example/*.txt",
                    help="Input file glob.")
flags.DEFINE_string("save_dir", "proc_data/example",
                    help="Directory for saving the processed data.")
flags.DEFINE_enum("split", "train", ["train", "valid", "test"],
                  help="Save the data as which split.")

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
  data_prefix = "data.{}".format(FLAGS.split)
  data_suffix = "tfrecord-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  tfrecord_name = format_filename(
      prefix=data_prefix,
      suffix=data_suffix,
      uncased=FLAGS.uncased,
  )
  tfrecord_path = os.path.join(save_dir, tfrecord_name)

  return tfrecord_path


def _meta_path(save_dir):
  """Get meta path."""
  meta_prefix = "meta.{}".format(FLAGS.split)
  meta_suffix = "json-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  meta_name = format_filename(
      prefix=meta_prefix,
      suffix=meta_suffix,
      uncased=FLAGS.uncased,
  )
  meta_path = os.path.join(save_dir, meta_name)

  return meta_path


def _create_data(input_paths, src_tok, tgt_tok):
  """Load data and call corresponding create_func."""
  num_src_tok, num_tgt_tok = 0, 0
  num_example = 0

  ##### Create record writer
  tfrecord_path = _tfrecord_path(FLAGS.save_dir)
  record_writer = tf.io.TFRecordWriter(tfrecord_path)
  tf.logging.info("Start writing tfrecord to %s.", tfrecord_path)

  for input_path in input_paths:
    line_cnt = 0

    tf.logging.info("Start processing %s", input_path)
    for line in tf.io.gfile.GFile(input_path, "r"):
      if line_cnt % 100000 == 0:
        tf.logging.info("Loading line %d", line_cnt)

      line = line.strip()
      if line:
        src, tgt = line.split("\t")

        src_ids = src_tok.convert_text_to_ids(src)
        tgt_ids = tgt_tok.convert_text_to_ids(tgt)

        # fairseq compatible
        src_ids = src_ids + [src_tok.eos_id]
        tgt_ids = [tgt_tok.eos_id] + tgt_ids

        # monitor stats
        num_src_tok += len(src_ids)
        num_tgt_tok += len(tgt_ids)

        feature = {
            "source": _int64_feature(src_ids),
            "target": _int64_feature(tgt_ids),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())

        line_cnt += 1
        num_example += 1

    tf.logging.info("Finish %s with %d lines.", input_path, line_cnt)
  record_writer.close()

  tf.logging.info("[Task %d] #examples: %d; #tokens: src %d, tgt %d",
                  FLAGS.task, num_example, num_src_tok, num_tgt_tok)

  ##### dump record information
  meta_info = {
      "filenames": [os.path.basename(tfrecord_path)],
      "num_example": num_example
  }
  meta_path = _meta_path(FLAGS.save_dir)
  with tf.io.gfile.GFile(meta_path, "w") as fp:
    json.dump(meta_info, fp)


def main(_):
  """create pretraining data (tfrecords)."""
  # Load tokenizer
  tokenizer = get_tokenizer()

  # Make workdirs
  if not tf.io.gfile.exists(FLAGS.save_dir):
    tf.io.gfile.makedirs(FLAGS.save_dir)

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

  _create_data(task_file_paths, src_tok=tokenizer, tgt_tok=tokenizer)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
