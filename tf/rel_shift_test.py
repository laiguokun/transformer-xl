import tensorflow as tf
from common_ops import rel_shift

tf.enable_eager_execution()

bsz = 1
seq_len = 16
chunk_len = 4

uni_inp = tf.range(seq_len, -1, -1)

bi_inp = tf.range(seq_len, -seq_len, -1)
bi_inp = bi_inp[seq_len - 2 * chunk_len:seq_len + 2 * chunk_len]
attn = tf.broadcast_to(bi_inp[None], [chunk_len, tf.shape(bi_inp)[0]])
print(attn)
shift_attn = rel_shift(attn, 0, chunk_len * 3)
print(shift_attn)

uni_inp = uni_inp[-2 * chunk_len - 1:]
attn = tf.broadcast_to(uni_inp[None], [chunk_len, tf.shape(uni_inp)[0]])
print(attn)
shift_attn = rel_shift(attn, 0, chunk_len * 2)
print(shift_attn)

