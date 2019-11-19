from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  import google3.experimental.users.zihangd.pretrain.seq2seq_input_func_builder as input_func_builder
  from google3.experimental.users.zihangd.pretrain.tokenization import get_tokenizer
except ImportError as e:
  print(e)
  import tensorflow as tf
  import seq2seq_input_func_builder as input_func_builder
  from tokenization import get_tokenizer
# pylint: enable=g-import-not-at-top


# tf.enable_eager_execution()

# Experiment (data/checkpoint/directory) paramenters
flags.DEFINE_string("record_dir", default="./proc_data/example/tfrecords",
                    help="Path to local directory containing *.tfrecord.")
flags.DEFINE_integer("bsz_per_host", default=512, help="batch size per host.")
flags.DEFINE_integer("max_length", default=256,
                     help="tgt len; 0 for not using it")
flags.DEFINE_string("split", default="train",
                    help="Data split.")
flags.DEFINE_integer("num_core_per_host", default=1, help="num core per host")
flags.DEFINE_integer("num_threads", default=1,
                     help="Num of passes to use.")
flags.DEFINE_integer("num_example", default=1,
                     help="Num of examples to see.")
flags.DEFINE_string("seq2seq_type", default="encdec",
                    help="encdec or dec_only")
flags.DEFINE_boolean("rel_attn", default=False,
                     help="whether to use rel attention")
FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv  # Unused

  tokenizer = get_tokenizer()

  input_fn, _ = input_func_builder.get_input_fn(
      tfrecord_dir=FLAGS.record_dir,
      split=FLAGS.split,
      max_length=FLAGS.max_length,
      num_hosts=1,
      uncased=FLAGS.uncased,
      num_threads=FLAGS.num_threads,
  )

  bsz_per_core = FLAGS.bsz_per_host // FLAGS.num_core_per_host
  params = {
      "batch_size": bsz_per_core
  }

  dataset = input_fn(params)
  example = dataset.make_one_shot_iterator().get_next()

  with tf.Session() as sess:
    for _ in range(FLAGS.num_example):
      example_np = sess.run(example)
      print("=" * 160)
      for k, v in example_np.items():
        print(k, v.shape)
        if v.ndim == 2:
          for i in range(v.shape[0]):
            if k in ["source", "target", "inputs", "targets"]:
              print(tokenizer.convert_ids_to_tokens(v[i].tolist()))
            else:
              print(v[i].tolist())
        elif v.ndim == 1:
          if k in ["source", "target", "inputs", "targets"]:
            print(tokenizer.convert_ids_to_tokens(v.tolist()))
          else:
            print(v.tolist())

if __name__ == "__main__":
  tf.app.run()
