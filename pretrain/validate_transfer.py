"""Perform pretraining."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  import google3.experimental.users.zihangd.pretrain.model_func_builder as model_func_builder
  from google3.experimental.users.zihangd.pretrain.tokenization import get_tokenizer
except ImportError as e:
  import tensorflow as tf
  import model_func_builder
  from tokenization import get_tokenizer
# pylint: enable=g-import-not-at-top


##### Ckpt to examine
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model.")

##### TPU
flags.DEFINE_bool("use_tpu", default=False,
                  help="Use TPUs rather than plain CPUs.")

##### Precision
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")
flags.DEFINE_bool("float32_softmax", default=True,
                  help="Whether to use float32 softmax.")

FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  #### Tokenizer
  tokenizer = get_tokenizer()

  #### Get corpus info
  n_token = tokenizer.get_vocab_size()
  tf.logging.info("n_token %d", n_token)

  # test data
  inputs_np = [3933, 7752, 15179, 893, 24249, 703, 19119, 4, 2919, 335, 8511,
               1094, 43, 1661, 669, 5481, 1106, 7029, 891, 891]
  type_id_np = [0] * len(inputs_np)
  inputs_np = np.array(inputs_np)[None]
  type_id_np = np.array(type_id_np)[None]

  # tensorflow graph
  inputs = tf.placeholder(tf.int64, [1, None])
  type_id = tf.placeholder(tf.int64, [1, None])
  hiddens = model_func_builder.extract_hiddens(
      inputs, type_id, n_token, is_training=False)

  # run session
  saver = tf.train.Saver()
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGS.init_checkpoint)

    feed_dict = {
        inputs: inputs_np,
        type_id: type_id_np,
    }

    hiddens_np = sess.run(hiddens, feed_dict=feed_dict)
    tf.logging.info(len(hiddens_np))


if __name__ == "__main__":
  app.run(main)
