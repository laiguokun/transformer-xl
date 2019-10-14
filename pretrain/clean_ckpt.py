"""Clean a ckpt to remove optimizer states."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import six
from six.moves import zip

import tensorflow as tf


flags.DEFINE_string("input_ckpt", "",
                    help="input ckpt for cleaning")
flags.DEFINE_string("output_model_dir", "",
                    help="output dir for cleaned ckpt")

FLAGS = flags.FLAGS


def clean_ckpt(_):
  """Core function."""
  input_ckpt = FLAGS.input_ckpt
  output_model_dir = FLAGS.output_model_dir

  tf.reset_default_graph()

  tf.logging.info("Loading from %s", input_ckpt)
  var_list = tf.contrib.framework.list_variables(input_ckpt)
  reader = tf.contrib.framework.load_checkpoint(input_ckpt)
  var_values, var_dtypes = {}, {}

  for (name, _) in var_list:
    if name.startswith("global_step") or "adam" in name.lower():
      continue
    tensor = reader.get_tensor(name)

    var_dtypes[name] = tensor.dtype
    var_values[name] = tensor

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  grouped_assign_op = tf.group(*assign_ops)
  global_step = tf.Variable(
      0, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  if not tf.gfile.Exists(output_model_dir):
    tf.gfile.MakeDirs(output_model_dir)

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    feed_dict = {}
    for p, (name, value) in zip(placeholders,
                                six.iteritems(var_values)):
      feed_dict[p] = value

    sess.run(grouped_assign_op, feed_dict)

    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, os.path.join(output_model_dir, "model.ckpt"),
               global_step=global_step)


if __name__ == "__main__":
  tf.app.run(clean_ckpt)
