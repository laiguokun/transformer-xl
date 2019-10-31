from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

import tensorflow as tf
tf.enable_eager_execution()

def get_type_id(tgt_len, tgt_idx, type_val):
  tgt_idx_left_shift = tgt_idx[:-1]
  type_val_right_shift = type_val[1:]
  new_type_id_shift = tf.scatter_nd(
    shape=[tgt_len],
    indices=tgt_idx_left_shift[:, None],
    updates=type_val_right_shift
  )
  new_type_id_shift = tf.concat([type_val[:1], new_type_id_shift], axis=0)
  new_type_id_shift = tf.math.cumsum(new_type_id_shift, exclusive=True)[1:]
  new_type_id = tf.scatter_nd(
    shape=[tgt_len],
    indices=tgt_idx[:, None],
    updates=type_val
  )
  new_type_id = tf.math.cumsum(new_type_id, exclusive=True)
  new_type_id = new_type_id_shift - new_type_id
  return new_type_id

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  seq_len = 16
  inputs = tf.range(1, seq_len + 1, 1, dtype=tf.int32)
  type_id = tf.range(1, seq_len + 1, 1, dtype=tf.int32)

  del_ratio = 0.1
  add_ratio = 0.1
  rep_ratio = 0.2

  rep_label = -1
  add_label = -2
  del_label = -3

  del_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  del_mask = del_rand < del_ratio
  non_del_mask = tf.logical_not(del_mask)

  right_shift_del_mask = tf.concat(
      [tf.constant(False, shape=[1]), del_mask[:-1]], axis=0)
  non_add_mask = tf.logical_or(del_mask, right_shift_del_mask)

  add_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  add_num = tf.reduce_sum(tf.cast(add_rand < add_ratio, tf.int32))
  add_uniform = tf.random.uniform(shape=[add_num, seq_len], minval=0, maxval=1)
  add_uniform -= 1e5 * tf.cast(non_add_mask, tf.float32)
  add_idx = tf.argmax(add_uniform, axis=1)
  add_cnt = tf.reduce_sum(tf.one_hot(add_idx, seq_len, dtype=tf.int32), 0)

  rep_rand = tf.random.uniform(shape=[seq_len], minval=0, maxval=1)
  rep_mask = tf.logical_and(tf.equal(add_cnt, 0), tf.logical_not(non_add_mask))
  rep_mask = tf.logical_and(
      rep_rand < (rep_ratio / (1 - 2 * del_ratio - add_ratio)), rep_mask)
  rep_input = tf.where(
      rep_mask,
      tf.constant(rep_label, shape=[seq_len]),
      inputs)

  tgt_len_encoder = tgt_len_decoder = seq_len
  print("rep", tf.cast(rep_mask, tf.int32).numpy().tolist())
  print("add", add_cnt.numpy().tolist())
  print("del", tf.cast(del_mask, tf.int32).numpy().tolist())

  ori_idx = tf.range(seq_len)

  #### encoder input
  shift_val = add_cnt - tf.cast(del_mask, tf.int32)
  shift_val = tf.cumsum(shift_val)

  shift_idx = ori_idx + shift_val

  tgt_len = tgt_len_encoder
  valid_tgt = shift_idx < tgt_len

  # remove deleted token
  tgt_idx = tf.boolean_mask(shift_idx, tf.logical_and(non_del_mask, valid_tgt))
  tgt_val = tf.boolean_mask(rep_input, tf.logical_and(non_del_mask, valid_tgt))
  type_val = tf.boolean_mask(type_id, tf.logical_and(non_del_mask, valid_tgt))

  max_len = tf.math.reduce_max(tgt_idx) + 1
  
  enc_type_id = get_type_id(tgt_len, tgt_idx, type_val)

  enc_seq = tf.scatter_nd(
      shape=[tgt_len],
      indices=tf.range(0, max_len)[:, None],
      updates=tf.zeros(shape=[max_len], dtype=tf.int32) + add_label
  )
  enc_seq = tf.tensor_scatter_nd_update(
      enc_seq,
      indices=tgt_idx[:, None],
      updates=tgt_val)

  print("encoder input")
  print(enc_seq.numpy().tolist())
  print(enc_type_id.numpy().tolist())

  #### decoder
  shift_val = tf.cumsum(add_cnt)
  shift_idx = ori_idx + shift_val

  tgt_len = tgt_len_decoder
  valid_tgt = shift_idx < tgt_len

  tgt_idx = tf.boolean_mask(shift_idx, valid_tgt)
  tgt_val = tf.boolean_mask(inputs, valid_tgt)
  type_val = tf.boolean_mask(type_id, valid_tgt)

  max_len = tf.math.reduce_max(tgt_idx) + 1

  pad_id = 100
  eos_id = 101
  add_id = 102

  dec_type_id = get_type_id(tgt_len, tgt_idx, type_val)

  dec_seq = tf.concat(
      [tf.zeros(shape=[max_len], dtype=tf.int32) + add_id,
       tf.zeros(shape=[tgt_len - max_len], dtype=tf.int32) + pad_id], 0)
  dec_seq = tf.tensor_scatter_nd_update(
      dec_seq,
      indices=tgt_idx[:, None],
      updates=tgt_val)

  # decoder input
  dec_inp = tf.concat([tf.constant(eos_id, shape=[1]), dec_seq[:-1]], 0)

  # edit type label
  dec_add_mask = tf.equal(dec_seq, add_id)
  dec_rep_mask = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx[:, None],
      updates=tf.boolean_mask(rep_mask, valid_tgt)
  )
  dec_del_mask = tf.scatter_nd(
      shape=[tgt_len],
      indices=tgt_idx[:, None],
      updates=tf.boolean_mask(del_mask, valid_tgt)
  )
  edit_label = tf.cast(dec_add_mask, tf.int32) * add_label
  edit_label += tf.cast(dec_rep_mask, tf.int32) * rep_label
  edit_label += tf.cast(dec_del_mask, tf.int32) * del_label

  print("decoder")
  print("inputs", dec_inp.numpy().tolist())
  print("target", dec_seq.numpy().tolist())
  print("labels", edit_label.numpy().tolist())
  print("type_id", dec_type_id.numpy().tolist())


if __name__ == "__main__":
  app.run(main)
