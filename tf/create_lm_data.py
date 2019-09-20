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
  import google3.experimental.users.zihangd.lm.data_utils as data_utils
  import google3.experimental.users.zihangd.lm.tokenization as tokenization
  import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

  run_internal = True
except ImportError as e:
  import tensorflow as tf
  import data_utils
  import tokenization

  run_internal = False
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS

flags.DEFINE_integer("seq_len", 512,
                     help="Sequence length.")
flags.DEFINE_integer("bsz_per_host", 32,
                     help="Batch size per host.")
flags.DEFINE_string("tokenizer_type", "sent_piece",
                    help="Type of the tokenizer.")
flags.DEFINE_string("tokenizer_paths", "",
                    help="Comma separated string.")
flags.DEFINE_bool("lower_case", True,
                  help="Use lower cased inputs or not.")
flags.DEFINE_bool("add_eos", False,
                  help="whether to append EOS at the end of a line.")
flags.DEFINE_bool("add_double_eos", False,
                  help="whether to append EOS at the begin and end of a line.")
flags.DEFINE_bool("from_raw_text", True,
                  help="Whether the input is raw text or encoded ids.")

flags.DEFINE_string("input_glob", "data/example/*.txt",
                    help="Input file glob.")
flags.DEFINE_string("save_dir", "proc_data/example",
                    help="Directory for saving the processed data.")
flags.DEFINE_enum("split", "train", ["train", "dev", "test"],
                  help="Save the data as which split.")
flags.DEFINE_integer("pass_id", 0, help="ID of the current pass."
                     "Different passes sample different concat orders.")

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


def _create_data(input_paths, tokenizer):
  """Load data and call corresponding create_func."""

  input_shards = []
  total_line_cnt = 0
  for input_path in input_paths:
    input_data = []
    line_cnt = 0
    tf.logging.info("===== Processing %s =====", input_path)
    for line in tf.gfile.Open(input_path):
      if line_cnt % 100000 == 0:
        tf.logging.info("Loading line %d", line_cnt)

      cur_sent = tokenizer.convert_text_to_ids(line.strip())
      if line_cnt < 5:
        tf.logging.debug(tokenizer.convert_text_to_tokens(line.strip()))

      if FLAGS.add_double_eos:
        cur_sent = [tokenizer.eos_id] + cur_sent + [tokenizer.eos_id]
      elif FLAGS.add_eos:
        cur_sent = cur_sent + [tokenizer.eos_id]

      if not cur_sent:
        tf.logging.info("Skip line %s --> tokens %s", line.strip(), cur_sent)
        continue

      input_data.extend(cur_sent)
      line_cnt += 1

    if line_cnt == 0:
      tf.logging.info("Skip empty file %s", input_path)
      continue
    else:
      tf.logging.info("Finish %s with %d lines", input_path, line_cnt)

    input_data = np.array(input_data, dtype=np.int64)

    total_line_cnt += line_cnt
    input_shards.append(input_data)

  tf.logging.info("[Task %d] Total number line: %d", FLAGS.task, total_line_cnt)

  filenames, num_batch = [], 0

  # Randomly shuffle input shards (with a fixed but unique random seed)
  np.random.seed(100 * FLAGS.task + FLAGS.pass_id)

  perm_indices = np.random.permutation(len(input_shards))
  tf.logging.info("Using perm indices %s for pass %d",
                  perm_indices.tolist(), FLAGS.pass_id)

  input_data_list = [input_shards[idx] for idx in perm_indices]
  input_data = np.concatenate(input_data_list)

  prefix = "data.{}.pass-{}".format(FLAGS.split, FLAGS.pass_id)
  suffix = "tfrecord-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  file_name = data_utils.format_filename(
      prefix=prefix,
      suffix=suffix,
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      lower_case=FLAGS.lower_case,
  )
  save_path = os.path.join(FLAGS.save_dir, file_name)

  cur_num_batch = create_lm_tfrecords(
      save_path=save_path,
      data=input_data,
  )

  filenames.append(file_name)
  num_batch += cur_num_batch

  record_info = {
      "filenames": filenames,
      "num_batch": num_batch,
  }

  return record_info


def create_data(_):
  """create pretraining data (tfrecords)."""
  tokenizer = tokenization.get_tokenizer(
      tokenizer_type=FLAGS.tokenizer_type,
      paths=FLAGS.tokenizer_paths.split(","),
      do_lower_case=FLAGS.lower_case)
  data_utils.setup_special_ids(tokenizer)

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

  record_info = _create_data(task_file_paths, tokenizer)

  prefix = "meta.{}.pass-{}".format(FLAGS.split, FLAGS.pass_id)
  suffix = "json-{:05d}-of-{:05d}".format(FLAGS.task, FLAGS.num_task)
  record_name = data_utils.format_filename(
      prefix=prefix,
      suffix=suffix,
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      lower_case=FLAGS.lower_case,
  )
  record_info_path = os.path.join(FLAGS.save_dir, record_name)

  with tf.gfile.Open(record_info_path, "w") as fp:
    json.dump(record_info, fp)


def batchify(data, bsz_per_host):
  """Turn flat data into proper batched format."""
  num_step = len(data) // bsz_per_host
  remain_len = bsz_per_host * num_step
  remove_len = len(data) - remain_len

  if remove_len > 0:
    # Randomly select begin step
    np.random.seed(100 * FLAGS.task + FLAGS.pass_id)
    beg_step = np.random.choice(remove_len, 1).item()
  else:
    beg_step = 0

  data = data[beg_step:beg_step + remain_len]
  data = data.reshape(bsz_per_host, num_step)

  return data


def create_lm_tfrecords(save_path, data):
  """create language modeling tfrecords from numpy array."""
  tf.logging.info("===== Create LM tfrecords =====")

  data = batchify(data, FLAGS.bsz_per_host)
  tf.logging.info("Raw data shape %s.", data.shape)

  record_writer = tf.python_io.TFRecordWriter(save_path)
  tf.logging.info("Start writing %s.", save_path)

  data_len = data.shape[1]

  i = 0
  num_batch = 0
  while i + FLAGS.seq_len <= data_len - 1:
    if num_batch % 500 == 0:
      tf.logging.info("Processing batch %d.", num_batch)

    all_ok = True
    features = []
    for idx in range(FLAGS.bsz_per_host):
      inp_id = data[idx, i: i + FLAGS.seq_len]
      tgt_id = data[idx, i + 1: i + FLAGS.seq_len + 1]

      feature = {
          "inputs": _int64_feature(inp_id),
          "target": _int64_feature(tgt_id),
      }
      features.append(feature)

    if all_ok:
      assert len(features) == FLAGS.bsz_per_host
      for feature in features:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
      num_batch += 1
    else:
      break

    i += FLAGS.seq_len

  record_writer.close()
  tf.logging.info("Done writing %s. Num of batches: %d", save_path, num_batch)

  return num_batch


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(create_data)
