from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

import tensorflow as tf
tf.enable_eager_execution()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

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
  add_num = tf.math.reduce_sum(tf.cast(add_rand < add_ratio, tf.int32)).numpy()
  non_add_logit = tf.math.log(tf.cast(tf.logical_not(non_add_mask), tf.float32))
  add_idx = tf.random.categorical(tf.expand_dims(non_add_logit, 0), add_num)
  add_idx = tf.squeeze(add_idx, 0)
  add_mask = tf.scatter_nd(
      shape=[seq_len],
      indices=add_idx[:, None],
      updates=tf.constant(1, shape=[add_num])
  )
  add_mask_bool = add_mask > 0
  
  rep_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  rep_mask = tf.logical_and(tf.logical_not(add_mask_bool), tf.logical_not(non_add_mask))
  rep_mask = tf.logical_and(
      rep_rand < (rep_ratio / (1 - 2 * del_ratio - add_ratio)), rep_mask)
  rep_input = tf.where(
      rep_mask,
      tf.constant(-1, shape=[seq_len]),
      inputs)

  tgt_len_encoder = tgt_len_decoder = seq_len
  print("rep", tf.cast(rep_mask, tf.int32).numpy().tolist())
  print("add", tf.cast(add_mask, tf.int32).numpy().tolist())
  print("del", tf.cast(del_mask, tf.int32).numpy().tolist())
  #### encoder input
  shift_val = add_mask - tf.cast(del_mask, tf.int32)
  shift_val = tf.cumsum(shift_val)

  ori_idx = tf.range(seq_len)
  shift_idx = ori_idx + shift_val

  tgt_len = tgt_len_encoder
  valid_tgt = shift_idx < tgt_len
  #### remove deleted token
  tgt_idx = tf.boolean_mask(shift_idx, tf.logical_and(non_del_mask, valid_tgt))
  tgt_val = tf.boolean_mask(rep_input, tf.logical_and(non_del_mask, valid_tgt))
  
  max_len = tf.math.reduce_max(tgt_idx).numpy() + 1

  output_encoder = tf.scatter_nd(
      shape=[tgt_len],
      indices=tf.range(0, max_len)[:, None],
      updates=tf.constant(-2, shape=[max_len])
  )
  output_encoder = tf.tensor_scatter_nd_update(
      output_encoder,
      indices=tgt_idx[:, None],
      updates=tgt_val)

  print("encoder input")
  print(output_encoder.numpy().tolist())
  #decoder output
  shift_val = tf.cumsum(add_mask)
  ori_idx = tf.range(seq_len)
  shift_idx = ori_idx + shift_val

  tgt_len = tgt_len_decoder
  valid_tgt = shift_idx < tgt_len

  tgt_idx = tf.boolean_mask(shift_idx, valid_tgt)
  tgt_val = tf.boolean_mask(inputs, valid_tgt)
  del_val = tf.boolean_mask(del_mask, valid_tgt)
  rep_val = tf.boolean_mask(rep_mask, valid_tgt)

  max_len = tf.math.reduce_max(tgt_idx).numpy() + 1

  output_decoder = tf.scatter_nd(
      shape=[tgt_len],
      indices=tf.range(0, max_len)[:, None],
      updates=tf.constant(-2, shape=[max_len])
  )
  output_decoder = tf.tensor_scatter_nd_update(
      output_decoder,
      indices=tgt_idx[:, None],
      updates=tgt_val)

  add_mask_decoder = tf.math.equal(output_decoder, tf.constant(-2, shape=[tgt_len]))
  rep_mask_decoder = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx[:, None],
      updates=rep_val
  )
  del_mask_decoder = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx[:, None],
      updates=del_val
  )

  print("decoder output")
  print(output_decoder.numpy().tolist())
  print("rep decoder", tf.cast(rep_mask_decoder, tf.int32).numpy().tolist())
  print("add decoder", tf.cast(add_mask_decoder, tf.int32).numpy().tolist())
  print("del decoder", tf.cast(del_mask_decoder, tf.int32).numpy().tolist())
  

if __name__ == '__main__':
  app.run(main)
