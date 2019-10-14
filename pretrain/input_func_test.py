from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  import google3.experimental.users.zihangd.pretrain.input_func_builder as input_func_builder
  from google3.experimental.users.zihangd.pretrain.tokenization import get_tokenizer
except ImportError as e:
  print(e)
  import tensorflow as tf
  import input_func_builder as input_func_builder
  from tokenization import get_tokenizer
# pylint: enable=g-import-not-at-top


# tf.enable_eager_execution()

# Experiment (data/checkpoint/directory) paramenters
flags.DEFINE_string("record_info_dir", default="./proc_data/example/tfrecords",
                    help="Path to local directory containing *.tfrecord.")
flags.DEFINE_integer("bsz_per_host", default=32, help="batch size per host.")
flags.DEFINE_integer("seq_len", default=512,
                     help="tgt len; 0 for not using it")
flags.DEFINE_integer("num_predict", default=85,
                     help="Num of tokens to predict.")
flags.DEFINE_string("split", default="train",
                    help="Data split.")
flags.DEFINE_integer("num_core_per_host", default=16, help="num core per host")
flags.DEFINE_integer("num_passes", default=None,
                     help="Num of passes to use.")
flags.DEFINE_string("loss_type", "mlm", help="")

FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv  # Unused

  tokenizer = get_tokenizer()

  input_fn, _ = input_func_builder.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split=FLAGS.split,
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      num_hosts=1,
      num_core_per_host=FLAGS.num_core_per_host,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      num_predict=FLAGS.num_predict,
      loss_type=FLAGS.loss_type
  )

  bsz_per_core = FLAGS.bsz_per_host // FLAGS.num_core_per_host
  params = {
      "batch_size": bsz_per_core
  }

  dataset = input_fn(params)
  example = dataset.make_one_shot_iterator().get_next()

  with tf.Session() as sess:
    for _ in range(20):
      example_np = sess.run(example)
      print("=" * 160)
      if FLAGS.loss_type in ["mlm"]:
        masked_input = tokenizer.convert_ids_to_tokens(
            example_np["masked_input"][0].tolist())
        print(masked_input)
      # print(example_np["label"][0])
      # print(example_np["seg_id"][0])

if __name__ == "__main__":
  tf.app.run()
