"""Common operations used to construct model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

import tensorflow as tf


FLAGS = flags.FLAGS


INF = 1e6


def safe_precision(func):
  @functools.wraps(func)
  def wrapped_func(inputs, *args, **kwargs):
    if inputs.dtype != tf.float32:
      fp32_inputs = tf.cast(inputs, tf.float32)
    else:
      fp32_inputs = inputs

    output = func(fp32_inputs, *args, **kwargs)

    if output.dtype != inputs.dtype:
      output = tf.cast(output, inputs.dtype)

    return output

  return wrapped_func


def get_einsum_prefix(ndims):
  einsum_symbols = ["u", "v", "w", "x", "y", "z"]
  assert ndims <= len(einsum_symbols)
  einsum_prefix = ""
  for i in range(ndims):
    einsum_prefix += einsum_symbols[i]

  return einsum_prefix


def update_monitor_dict(tgt, src, prefix=None):
  if prefix is None:
    tgt.update(src)
  else:
    for k, v in src.items():
      tgt["{}/{}".format(prefix, k)] = v

  return tgt


def safe_softmax(inputs, *args, **kwargs):
  if getattr(FLAGS, "float32_softmax", True) and inputs.dtype != tf.float32:
    inputs_f32 = tf.cast(inputs, tf.float32)
    output = tf.nn.softmax(inputs_f32, *args, **kwargs)
    output = tf.cast(output, inputs.dtype)
  else:
    output = tf.nn.softmax(inputs, *args, **kwargs)

  return output


layer_norm = safe_precision(tf.contrib.layers.layer_norm)


def sinusoid_positional_encoding(seq_len, d_model, pos_seq=None, bsz=None,
                                 clamp_len=-1, dtype=tf.float32):
  """create relative positional encoding."""
  assert d_model % 2 == 0, "`d_model` must be an even number."

  if pos_seq is None:
    pos_seq = tf.range(0, seq_len, 1.0)[None]

  half_dim = d_model // 2
  freq_seq = tf.cast(tf.range(0, half_dim, 1.0), dtype=dtype)
  if dtype is not None and dtype != tf.float32:
    freq_seq = tf.cast(freq_seq, dtype=dtype)
  inv_freq = 1 / (10000 ** (freq_seq / half_dim))

  # type cast
  if dtype is not None and pos_seq.dtype != dtype:
    pos_seq = tf.cast(pos_seq, dtype=dtype)

  # clamp maximum length
  if clamp_len > 0:
    pos_seq = tf.clip_by_value(pos_seq, -clamp_len, clamp_len)

  sinusoid_inp = tf.einsum("bi,d->bid", pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

  if bsz is not None and pos_seq is None:
    length = tf.shape(pos_emb)[1]
    pos_emb = tf.broadcast_to(pos_emb, [bsz, length, d_model])

  return pos_emb


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_type):
  """Get the corresponding activation function from string."""
  if activation_type == "relu":
    activation = tf.nn.relu
  elif activation_type == "gelu":
    activation = gelu
  elif activation_type == "safe_gelu":
    activation = safe_precision(gelu)
  elif activation_type == "tanh":
    activation = safe_precision(tf.tanh)
  else:
    raise ValueError("Unsupported activation type {}".format(activation_type))

  return activation


def positionwise_ffn(inp, d_model, d_inner, dropout, dropact, initializer,
                     activation_type="relu", scope="ff", is_training=True,
                     reuse=None, use_residual=True):
  """Position-wise Feed-forward Network."""
  monitor_dict = {}

  activation = get_activation(activation_type)

  output = inp
  with tf.variable_scope(scope, reuse=reuse):
    output = tf.layers.dense(output, d_inner, activation=activation,
                             kernel_initializer=initializer,
                             name="layer_1")
    output = tf.layers.dropout(output, dropact, training=is_training,
                               name="drop_1")
    output = tf.layers.dense(output, d_model,
                             kernel_initializer=initializer,
                             name="layer_2")
    output = tf.layers.dropout(output, dropout, training=is_training,
                               name="drop_2")

    # post ffn process
    output, res_lnorm_dict = residual_and_layer_norm(
        inp, output, use_residual)
    monitor_dict = update_monitor_dict(monitor_dict, res_lnorm_dict)

  return output, monitor_dict


def residual_and_layer_norm(residual, hidden, use_residual):
  """Perform residual & layer normalization."""
  monitor_dict = {}

  if use_residual:
    output = hidden + residual
  else:
    output = hidden

  output = layer_norm(
      output, begin_norm_axis=-1, scope="LayerNorm")

  with tf.variable_scope("LayerNorm", reuse=True):
    monitor_dict["ln_gamma"] = tf.get_variable("gamma")
    monitor_dict["ln_beta"] = tf.get_variable("beta")

  monitor_dict["ln_inp_res"] = residual
  monitor_dict["ln_inp"] = hidden
  monitor_dict["ln_out"] = output

  return output, monitor_dict


def embedding_lookup(x, n_token, d_embed, initializer, lookup_table=None,
                     use_tpu=True, scope="embedding", reuse=None,
                     dtype=tf.float32):
  """tpu and gpu embedding_lookup function."""
  with tf.variable_scope(scope, reuse=reuse):
    if lookup_table is None:
      lookup_table = tf.get_variable("lookup_table", [n_token, d_embed],
                                     dtype=dtype, initializer=initializer)

    if use_tpu:
      one_hot_idx = tf.one_hot(x, n_token, dtype=dtype)
      einsum_prefix = get_einsum_prefix(x.shape.ndims)
      einsum_str = "{0}n,nd->{0}d".format(einsum_prefix)
      output = tf.einsum(einsum_str, one_hot_idx, lookup_table)
    else:
      output = tf.nn.embedding_lookup(lookup_table, x)

    return output, lookup_table


def causal_attn_mask(qlen, mlen=None, dtype=tf.float32):
  """create causal attention mask."""
  attn_mask = tf.ones([qlen, qlen], dtype=dtype)
  mask_up_tri = tf.matrix_band_part(attn_mask, 0, -1)
  mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
  ret = mask_up_tri - mask_dia
  if mlen is not None:
    attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
    ret = tf.concat([attn_mask_pad, ret], 1)

  return ret


def merge_attn_masks(attn_mask, input_mask, perm_mask, mlen=None):
  """merge attention masks into a single attention mask."""
  # input_mask [B x T] + perm_mask [B x F x T] -> data_mask [B x F x T]
  if input_mask is not None and perm_mask is not None:
    data_mask = input_mask[:, None, :] + perm_mask
  elif input_mask is not None and perm_mask is None:
    data_mask = input_mask[:, None, :]
  elif input_mask is None and perm_mask is not None:
    data_mask = perm_mask
  else:
    data_mask = None

  if data_mask is not None:
    # create memory mask: [B x F x T] -> [B x F x T']
    if mlen is not None:
      data_mask_shape = tf.shape(data_mask)
      # all mems can be attended to
      mems_mask = tf.zeros([data_mask_shape[0], data_mask_shape[1], mlen],
                           dtype=data_mask.dtype)
      data_mask = tf.concat([mems_mask, data_mask], 2)

    # data_mask [B x F x T'] + attn_mask [B x N x F x T'] -> [B x N x F x T']
    if attn_mask is None:
      attn_mask = data_mask[:, None, :, :]
    else:
      attn_mask += data_mask[:, None, :, :]

  # type cast
  if attn_mask is not None:
    attn_mask = tf.cast(attn_mask > 0, dtype=attn_mask.dtype)

  return attn_mask


def head_projection(h, d_model, n_head, d_head, kernel_initializer, name):
  """Project hidden states to a specific head with a 4D-shape."""
  proj_weight = tf.get_variable("{}/kernel".format(name),
                                [d_model, n_head, d_head], dtype=h.dtype,
                                initializer=kernel_initializer)
  einsum_prefix = get_einsum_prefix(h.shape.ndims - 1)
  einsum_str = "{0}h,hnd->{0}nd".format(einsum_prefix)
  head = tf.einsum(einsum_str, h, proj_weight)

  proj_bias = tf.get_variable("{}/bias".format(name),
                              [n_head, d_head], dtype=h.dtype,
                              initializer=tf.zeros_initializer())
  head += proj_bias

  return head


def post_attention(h, attn_vec, d_model, n_head, d_head, dropout, is_training,
                   kernel_initializer, residual=True):
  """Post-attention processing."""
  monitor_dict = {}

  # post-attention projection (back to `d_model`)
  proj_o = tf.get_variable("o/kernel", [d_model, n_head, d_head],
                           dtype=h.dtype, initializer=kernel_initializer)
  einsum_prefix = get_einsum_prefix(attn_vec.shape.ndims - 2)
  einsum_str = "{0}nd,hnd->{0}h".format(einsum_prefix)
  attn_out = tf.einsum(einsum_str, attn_vec, proj_o)
  proj_bias = tf.get_variable("o/bias",
                              [d_model], dtype=h.dtype,
                              initializer=tf.zeros_initializer())
  attn_out += proj_bias

  attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

  output, res_lnorm_dict = residual_and_layer_norm(
      h, attn_out, use_residual=residual)
  monitor_dict = update_monitor_dict(monitor_dict, res_lnorm_dict)

  return output, monitor_dict


def abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt, is_training,
                  scale):
  """Core absolute positional attention operations."""
  monitor_dict = {}

  attn_score = tf.einsum("bind,bjnd->bnij", q_head, k_head)
  attn_score *= scale
  if attn_mask is not None:
    tf.logging.info("Attention mask shape: %s", attn_mask.shape)
    attn_score = attn_score - INF * attn_mask

  # attention probability
  attn_prob = safe_softmax(attn_score, -1)
  monitor_dict["attn_prob"] = attn_prob
  attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

  # attention output
  attn_vec = tf.einsum("bnij,bjnd->bind", attn_prob, v_head)

  return attn_vec, monitor_dict


def multihead_attn(q, k, v, attn_mask, d_model, n_head, d_head, dropout,
                   dropatt, is_training, kernel_initializer, residual=True,
                   reuse=None, scope="abs_attn"):
  """Standard multi-head attention with absolute positional embedding."""
  monitor_dict = {}

  scale = 1 / (d_head ** 0.5)
  with tf.variable_scope(scope, reuse=reuse):
    # attention heads
    q_head = head_projection(
        q, d_model, n_head, d_head, kernel_initializer, "q")
    k_head = head_projection(
        k, d_model, n_head, d_head, kernel_initializer, "k")
    v_head = head_projection(
        v, d_model, n_head, d_head, kernel_initializer, "v")

    # attention vector
    attn_vec, attn_core_dict = abs_attn_core(
        q_head, k_head, v_head, attn_mask, dropatt, is_training, scale)

    # post processing
    output, post_dict = post_attention(
        v, attn_vec, d_model, n_head, d_head, dropout,
        is_training, kernel_initializer, residual)

    # things to monitor
  monitor_dict = update_monitor_dict(monitor_dict, attn_core_dict)
  monitor_dict = update_monitor_dict(monitor_dict, post_dict)

  return output, monitor_dict

