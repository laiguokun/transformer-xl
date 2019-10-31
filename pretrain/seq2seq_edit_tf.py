from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

import tensorflow as tf
tf.enable_eager_execution()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  seq_len = 16
  inputs = tf.range(1, seq_len + 1, 1, dtype=tf.int32)

  del_ratio = 0.1
  add_ratio = 0.1
  rep_ratio = 0.1

  del_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  del_mask = del_rand < del_ratio
  non_del_mask = tf.logical_not(del_mask)

  right_shift_del_mask = tf.concat(
      [tf.constant(False, shape=[1]), del_mask[:-1]], axis=0)
  non_add_mask = tf.logical_or(
      del_mask, right_shift_del_mask)
  add_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  add_mask = tf.logical_and(
      add_rand < (add_ratio / (1 - del_ratio * 2)),
      tf.logical_not(non_add_mask))

  print(del_mask)
  print(non_add_mask)
  print(add_mask)

  rep_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  rep_mask = tf.logical_and(tf.logical_not(add_mask), non_del_mask)
  rep_mask = tf.logical_and(
      rep_rand < (rep_ratio / (1 - del_ratio - add_ratio)), rep_mask)

  shift_val = tf.cast(add_mask, tf.int32) - tf.cast(del_mask, tf.int32)
  shift_val = tf.cumsum(shift_val)

  ori_idx = tf.range(seq_len)
  shift_idx = ori_idx + shift_val
  rep_input = tf.where(
      rep_mask,
      tf.constant(-1, shape=[seq_len]),
      inputs)

  tgt_len = seq_len
  valid_tgt = shift_idx < tgt_len

  tgt_idx = tf.boolean_mask(shift_idx, tf.logical_and(non_del_mask, valid_tgt))
  tgt_val = tf.boolean_mask(rep_input, tf.logical_and(non_del_mask, valid_tgt))

  valid_add = shift_idx - 1 < tgt_len
  add_idx = tf.boolean_mask(shift_idx - 1, tf.logical_and(add_mask, valid_add))
  add_val = tf.zeros(shape=tf.shape(add_idx), dtype=tf.int32) - 2

  output = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx[:, None],
      updates=tgt_val)
  print(output.numpy().tolist())
  output = tf.tensor_scatter_nd_update(
      output,
      indices=add_idx[:, None],
      updates=add_val)
  print(output.numpy().tolist())
  print("rep", tf.cast(rep_mask, tf.int32).numpy().tolist())
  print("add", tf.cast(add_mask, tf.int32).numpy().tolist())
  print("del", tf.cast(del_mask, tf.int32).numpy().tolist())


if __name__ == "__main__":
  app.run(main)
