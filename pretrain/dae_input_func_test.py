from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  # import google3.experimental.users.zihangd.pretrain.dae_input_func_builder as input_func_builder
  import google3.experimental.users.zihangd.pretrain.daespan_input_func_builder as input_func_builder
  from google3.experimental.users.zihangd.pretrain.tokenization import get_tokenizer
except ImportError as e:
  print(e)
  import tensorflow as tf
  # import dae_input_func_builder as input_func_builder
  import daespan_input_func_builder as input_func_builder
  from tokenization import get_tokenizer
# pylint: enable=g-import-not-at-top


# tf.enable_eager_execution()

# Experiment (data/checkpoint/directory) paramenters
flags.DEFINE_string("doc_dir", default="",
                    help="Path to directory containing doc tfrecord.")
flags.DEFINE_string("sent_dir", default="",
                    help="Path to directory containing sent tfrecord.")
flags.DEFINE_string("semi_dir", default="",
                    help="Path to directory containing semi-doc tfrecord.")
flags.DEFINE_integer("bsz_per_host", default=32, help="batch size per host.")
flags.DEFINE_integer("seq_len", default=512,
                     help="tgt len; 0 for not using it")
flags.DEFINE_string("split", default="train",
                    help="Data split.")
flags.DEFINE_integer("num_core_per_host", default=16, help="num core per host")
flags.DEFINE_integer("num_example", default=2,
                     help="Num of examples to see.")
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
FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv  # Unused

  tokenizer = get_tokenizer()

  input_fn = input_func_builder.get_input_fn(
      doc_dir=FLAGS.doc_dir,
      semi_dir=FLAGS.semi_dir,
      sent_dir=FLAGS.sent_dir,
      split=FLAGS.split,
      uncased=FLAGS.uncased,
      seq_len=FLAGS.seq_len,
      bsz_per_host=FLAGS.bsz_per_host,
      num_hosts=1,
      num_core_per_host=FLAGS.num_core_per_host,
  )

  bsz_per_core = FLAGS.bsz_per_host // FLAGS.num_core_per_host
  params = {
      "batch_size": bsz_per_core
  }

  dataset = input_fn(params)
  example = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  for k, v in example.items():
    print(k, v.shape)

  with tf.Session() as sess:
    for _ in range(FLAGS.num_example):
      example_np = sess.run(example)
      print("=" * 160)
      for k, v in example_np.items():
        if v.ndim == 2:
          for i in range(v.shape[0]):
            if k in ["gen_inp", "gen_tgt", "dec_inp", "dec_tgt", "inputs", "dec_masked_tgt"]:
              print(k, v[i].shape, tokenizer.convert_ids_to_text(v[i].tolist()))
            else:
              print(k, v[i].shape, " ".join([str(j) for j in v[i].tolist()]))
        elif v.ndim == 1:
          if k in ["gen_inp", "gen_tgt", "dec_inp", "dec_tgt", "inputs", "dec_masked_tgt"]:
            print(k, v.shape, tokenizer.convert_ids_to_text(v.tolist()))
          else:
            print(k, v.shape, " ".join([str(j) for j in v.tolist()]))
        elif v.ndim > 3:
          for i in range(v.shape[0]):
            print(k, v.shape, v[i])


if __name__ == "__main__":
  tf.app.run()
