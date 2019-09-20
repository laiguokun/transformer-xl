"""Common operations used to construct model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import functools

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

import tensorflow as tf

flags.DEFINE_bool("use_head_bias", default=False,
                  help="Whether to use bias in head projection.")
flags.DEFINE_bool("share_w", default=False,
                  help="Whether to share W_k and W_r.")
flags.DEFINE_bool("use_act_dropout", default=True,
                  help="Apply dropout to nonlinear activation in FF.")
flags.DEFINE_float("ln_eps", default=1e-12, help="")
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


def get_current_layer():
  """Extract layer id from scope name."""
  scope_name = tf.get_variable_scope().name

  # transformer layers
  m = re.search(r"model/transformer/layer_(\d+?)/", scope_name + "/")
  if m is not None:
    return int(m.group(1))

  # input layer
  m = re.search(r"model/transformer/input/", scope_name + "/")
  if m is not None:
    return -1

  return None


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


def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None):
  # https://github.com/pytorch/fairseq/blob/5d543f9b19e76772386903d4eeebdceaeb3d1b69/fairseq/modules/layer_norm.py#L9
  # https://github.com/NVIDIA/apex/blob/3ef01faef2492b3e650f44ecc510f3a8f2426783/csrc/layer_norm_cuda_kernel.cu#L303
  # https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/ops/nn_impl.py#L1240
  """Custom Layer Normalization with changable epsilon."""
  with tf.variable_scope(scope, 'LayerNorm', [inputs], reuse=reuse):
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                       'must be < rank(inputs) (%d)' %
                       (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
          (inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = tf.get_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=tf.zeros_initializer(),
          trainable=trainable)
    if scale:
      gamma = tf.get_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=tf.ones_initializer(),
          trainable=trainable)
    # By default, compute the moments across all the dimensions except the one
    # with index 0.
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    # Note that epsilon must be increased for float16 due to the limited
    # representable range.
    variance_epsilon = (FLAGS.ln_eps if dtype != tf.float16
                        else max(FLAGS.ln_eps, 1e-3))
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect."""
  if not is_training:
    return inputs

  # Compute keep_prob
  keep_prob = 1.0 - drop_connect_rate

  # Compute drop_connect tensor
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output


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


def erf_gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0)))
    return x * cdf


def get_activation(activation_type):
  """Get the corresponding activation function from string."""
  if activation_type == "relu":
    activation = tf.nn.relu
  elif activation_type == "gelu":
    activation = gelu
  elif activation_type == "safe_gelu":
    activation = safe_precision(gelu)
  elif activation_type == "erf_gelu":
    activation = erf_gelu
  elif activation_type == "tanh":
    activation = safe_precision(tf.tanh)
  else:
    raise ValueError("Unsupported activation type {}".format(activation_type))

  return activation


def positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer,
                     activation_type="gelu", scope="ff", is_training=True,
                     reuse=None, use_residual=True):
  """Position-wise Feed-forward Network."""
  monitor_dict = {}

  activation = get_activation(activation_type)

  output = inp
  with tf.variable_scope(scope, reuse=reuse):
    output = tf.layers.dense(output, d_inner, activation=activation,
                            kernel_initializer=kernel_initializer,
                            name="layer_1")
    if FLAGS.use_act_dropout:
      output = tf.layers.dropout(output, dropout, training=is_training,
                                 name="drop_1")

    output = tf.layers.dense(output, d_model,
                            kernel_initializer=kernel_initializer,
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


def embedding_lookup(x, n_token, d_embed, initializer, use_tpu=True,
                     scope="embedding", reuse=None, dtype=tf.float32):
  """tpu and gpu embedding_lookup function."""
  with tf.variable_scope(scope, reuse=reuse):
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


def positional_embedding(pos_seq, inv_freq, bsz=None):
  sinusoid_inp = tf.einsum("i,d->id", pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  pos_emb = pos_emb[None, :, :]

  if bsz is not None:
    pos_emb = tf.tile(pos_emb, [bsz, 1, 1])

  return pos_emb


def sinusoid_encoding(d_model, fwd_pos_seq, bwd_pos_seq=None, bsz=None,
                      clamp_len=-1, dtype=tf.float32):
  """create sinuoid based encoding for pretraining."""
  freq_seq = tf.cast(tf.range(0, d_model, 2.0), dtype=dtype)
  if dtype is not None and dtype != tf.float32:
    freq_seq = tf.cast(freq_seq, dtype=dtype)
  inv_freq = 1 / (10000 ** (freq_seq / d_model))

  if bwd_pos_seq is not None:
    if dtype is not None and dtype != tf.float32:
      fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
      bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

    if clamp_len > 0:
      fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
      bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -clamp_len, clamp_len)

    if bsz is not None:
      fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
      bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
    else:
      fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq)
      bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq)

    pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=0)
  else:
    if dtype is not None and dtype != tf.float32:
      fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
    if clamp_len > 0:
      fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
    pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz=bsz)

  return pos_emb


def absolute_positional_encoding(klen, d_model, bi_data, bsz=None, clamp_len=-1,
                                 dtype=tf.float32, sinusoid=True, max_len=None,
                                 initializer=None):
  """create absolute positional encoding."""

  if sinusoid:
    if bi_data:
      fwd_pos_seq = tf.range(0, klen, 1.0)
      bwd_pos_seq = tf.range(klen - 1, -1, -1.0)
    else:
      fwd_pos_seq = tf.range(0., klen, 1.0)
      bwd_pos_seq = None

    pos_emb = sinusoid_encoding(
        d_model=d_model,
        fwd_pos_seq=fwd_pos_seq,
        bwd_pos_seq=bwd_pos_seq,
        bsz=bsz,
        clamp_len=clamp_len,
        dtype=dtype)
  else:
    # trainable positional encoding
    assert max_len is not None and initializer is not None
    pos_embedding = tf.get_variable("pos_embedding", [max_len, d_model],
                                    initializer=initializer, dtype=dtype)
    per_batch_shape = [1, 1]

    # assert clamp_len == -1, "Do not support clamp_len yet."
    if bi_data:
      fwd_pos_emb = pos_embedding[:klen][None]
      bwd_pos_emb = tf.reverse(fwd_pos_emb, tf.constant(0, shape=[1]))
      if bsz is not None:
        fwd_pos_emb = tf.tile(fwd_pos_emb, [bsz//2] + per_batch_shape)
        bwd_pos_emb = tf.tile(bwd_pos_emb, [bsz//2] + per_batch_shape)
      pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=0)
    else:
      fwd_pos_emb = pos_embedding[:klen][None]
      if bsz is not None:
        fwd_pos_emb = tf.tile(fwd_pos_emb, [bsz] + per_batch_shape)
      pos_emb = fwd_pos_emb

  return pos_emb


def relative_positional_encoding(qlen, klen, d_model, bi_data, attn_type,
                                 bsz=None, clamp_len=-1, dtype=tf.float32,
                                 sinusoid=True, max_len=None, n_head=None,
                                 d_head=None, initializer=None):

  """create relative positional encoding."""
  if attn_type in ["bi"]:
    beg, end = klen, -qlen
  else:
    beg, end = klen, -1

  if sinusoid:
    if bi_data:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      bwd_pos_seq = tf.range(-beg, -end, 1.0)
    else:
      fwd_pos_seq = tf.range(beg, end, -1.0)
      bwd_pos_seq = None

    pos_emb = sinusoid_encoding(
        d_model=d_model,
        fwd_pos_seq=fwd_pos_seq,
        bwd_pos_seq=bwd_pos_seq,
        bsz=bsz,
        clamp_len=clamp_len,
        dtype=dtype)
  else:
    # trainable positional encoding
    assert max_len is not None and initializer is not None

    if n_head is not None and d_head is not None:
      feat_shape = [n_head, d_head]
    else:
      feat_shape = [d_model]

    pos_embedding = tf.get_variable("rel_pos_embedding",
                                    [2 * max_len + 1] + feat_shape,
                                    initializer=initializer, dtype=dtype)

    if bi_data:
      fwd_pos_emb = pos_embedding[max_len - beg: max_len - end][None]
      bwd_pos_emb = tf.reverse(
          pos_embedding[max_len + end: max_len + beg],
          tf.constant(0, shape=[1]))[None]
      if bsz is not None:
        fwd_pos_emb = tf.tile(fwd_pos_emb, [bsz//2, 1] + [1] * len(feat_shape))
        bwd_pos_emb = tf.tile(bwd_pos_emb, [bsz//2, 1] + [1] * len(feat_shape))
      pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=0)
    else:
      fwd_pos_emb = pos_embedding[max_len - klen:max_len + qlen][None]
      if bsz is not None:
        fwd_pos_emb = tf.tile(fwd_pos_emb, [bsz, 1] + [1] * len(feat_shape))
      pos_emb = fwd_pos_emb

  return pos_emb


def causal_attn_mask(qlen, mlen, dtype=tf.float32, same_length=False):
  """create causal attention mask."""
  attn_mask = tf.ones([qlen, qlen], dtype=dtype)
  mask_u = tf.matrix_band_part(attn_mask, 0, -1)
  mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
  attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
  ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
  if same_length:
    mask_l = tf.matrix_band_part(attn_mask, -1, 0)
    ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

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


def local_rel_embed(full_rel_embed, chunk_len, attn_type, seq_len=None):
  if attn_type == "uni":
    return full_rel_embed[:, -2 * chunk_len - 1:]
  elif attn_type == "bi":
    assert seq_len is not None
    return full_rel_embed[:,
        seq_len - 2 * chunk_len - 1:seq_len + 2 * chunk_len]
  else:
    raise NotImplementedError


def local_attn_mask(input_mask, chunk_len):
  """Get local attention mask from input mask [bsz x seq_len]."""
  if input_mask is None:
    return None

  bsz = tf.shape(input_mask)[0]
  seq_len = tf.shape(input_mask)[1]

  if chunk_len == -1:
    # [B x T] -> [B x N x F x T]
    attn_mask = input_mask[:, None, None, :]
  else:
    # [B x C x T]
    input_mask = tf.reshape(
        input_mask, [bsz, seq_len // chunk_len, chunk_len])
    # [B x C x T]
    input_mask_l = tf.pad(
        input_mask[:, :-1], tf.constant([[0, 0], [1, 0], [0, 0]]))
    # [B x C x T]
    input_mask_r = tf.pad(
        input_mask[:, 1:], tf.constant([[0, 0], [0, 1], [0, 0]]))

    # [B x C x 3T]
    attn_mask = tf.concat([input_mask_l, input_mask, input_mask_r],
                          axis=-1)

    # [B x N x C x F x 3T]
    attn_mask = attn_mask[:, None, :, None, :]

  return attn_mask


def head_projection(h, d_model, n_head, d_head, kernel_initializer, name):
  """Project hidden states to a specific head with a 4D-shape."""
  proj_weight = tf.get_variable("{}/kernel".format(name),
                                [d_model, n_head, d_head], dtype=h.dtype,
                                initializer=kernel_initializer)
  einsum_prefix = get_einsum_prefix(h.shape.ndims - 1)
  einsum_str = "{0}h,hnd->{0}nd".format(einsum_prefix)
  head = tf.einsum(einsum_str, h, proj_weight)

  if FLAGS.use_head_bias:
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
  if FLAGS.use_head_bias:
    proj_bias = tf.get_variable("o/bias",
                                [d_model], dtype=h.dtype,
                                initializer=tf.zeros_initializer())
    attn_out += proj_bias

  attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

  output, res_lnorm_dict = residual_and_layer_norm(
      h, attn_out, use_residual=residual)
  monitor_dict = update_monitor_dict(monitor_dict, res_lnorm_dict)

  return output, monitor_dict


def rel_shift(x, row_dim, klen=-1):
  """Perform relative shift to form the relative attention score."""
  ndims = x.shape.ndims
  x_shape = tf.shape(x)

  # Deal with negative indexing
  if row_dim < 0:
    row_dim = ndims + row_dim
  assert row_dim >= 0

  # Assume `col_dim` = `row_dim + 1`
  col_dim = row_dim + 1
  assert col_dim < ndims

  tgt_shape_1, slice_begin_1, slice_len_1 = [], [], []
  tgt_shape_2, slice_begin_2, slice_len_2 = [], [], []
  for i in range(ndims):
    slice_len_1.append(-1)
    slice_begin_2.append(0)

    if i == row_dim:
      tgt_shape_1.append(x_shape[col_dim])
      tgt_shape_2.append(x_shape[row_dim])
      slice_begin_1.append(1)
      slice_len_2.append(-1)
    elif i == col_dim:
      tgt_shape_1.append(x_shape[row_dim])
      tgt_shape_2.append(x_shape[col_dim] - 1)
      slice_begin_1.append(0)
      slice_len_2.append(klen)
    else:
      tgt_shape_1.append(x_shape[i])
      tgt_shape_2.append(x_shape[i])
      slice_begin_1.append(0)
      slice_len_2.append(-1)

  x = tf.reshape(x, tgt_shape_1)
  x = tf.slice(x, slice_begin_1, slice_len_1)
  x = tf.reshape(x, tgt_shape_2)
  x = tf.slice(x, slice_begin_2, slice_len_2)

  return x


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


def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training,
                  scale):
  """Core relative positional attention operations."""
  monitor_dict = {}

  # content based attention score
  ac = tf.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head_h)

  # position based attention score
  bd = tf.einsum("bind,bjnd->bnij", q_head + r_r_bias, k_head_r)
  bd = rel_shift(bd, row_dim=2, klen=tf.shape(ac)[-1])

  # segment based attention score
  if seg_mat is None:
    ef = 0
  else:
    # seg_embed: [2 x N x D]
    ef = tf.einsum("bind,snd->bnis", q_head + r_s_bias, seg_embed)
    # seg_mat: [B x F x T]  -> [B x 1 x F x T]-> [B x N x F x T]
    tgt_shape = tf.shape(bd)
    ef = tf.where(tf.broadcast_to(tf.expand_dims(seg_mat, 1), tgt_shape),
                  tf.broadcast_to(ef[:, :, :, 1:], tgt_shape),
                  tf.broadcast_to(ef[:, :, :, :1], tgt_shape))

  # merge attention scores and perform masking
  attn_score = (ac + bd + ef) * scale
  if attn_mask is not None:
    tf.logging.info("Attention mask shape: %s", attn_mask.shape)
    attn_score = attn_score - 1e30 * attn_mask

  # attention probability
  attn_prob = safe_softmax(attn_score, -1)
  monitor_dict["attn_prob"] = attn_prob
  attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

  # attention output
  attn_vec = tf.einsum("bnij,bjnd->bind", attn_prob, v_head_h)

  # things to monitor in attention
  monitor_dict["attn_ac"] = ac
  monitor_dict["attn_bd"] = bd
  if seg_mat is not None:
    monitor_dict["attn_ef"] = ef

  return attn_vec, monitor_dict


def local_abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt, is_training,
                        scale):
  """Core absolute positional attention operations."""
  monitor_dict = {}

  attn_c = tf.einsum("bcxnh,bcynh->bncxy", q_head, k_head) * scale
  attn_l = tf.einsum("bcxnh,bcynh->bncxy",
                     q_head[:, 1:], k_head[:, :-1]) * scale
  attn_l = tf.pad(
      attn_l,
      tf.constant([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]]),
      constant_values=-INF)

  attn_score = tf.concat([attn_l, attn_c], axis=-1)
  if attn_mask is not None:
    tf.logging.info("Attention mask shape: %s", attn_mask.shape)
    attn_score = attn_score - INF * attn_mask
  attn_prob = safe_softmax(attn_score, axis=-1)
  monitor_dict["attn_prob"] = attn_prob
  attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
  attn_prob_l, attn_prob_c = tf.split(
      attn_prob,
      num_or_size_splits=2,
      axis=-1)

  # attention vector
  pad_v_head = tf.pad(
      v_head,
      tf.constant([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]])
  )
  attn_vec = (tf.einsum("bncxy,bcynh->bcxnh", attn_prob_l, pad_v_head[:, :-2]) +
              tf.einsum("bncxy,bcynh->bcxnh", attn_prob_c, v_head))

  return attn_vec, monitor_dict


def local_rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, r_w_bias,
                        r_r_bias, attn_mask, dropatt, is_training, scale):
  """Core local relative positional attention operations."""
  monitor_dict = {}

  # content based attention
  attn_c = tf.einsum("bcxnh,bcynh->bncxy", q_head + r_w_bias, k_head_h) * scale
  attn_l = tf.einsum("bcxnh,bcynh->bncxy",
                     q_head[:, 1:], k_head_h[:, :-1]) * scale
  attn_l = tf.pad(
      attn_l,
      tf.constant([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]]),
      constant_values=-INF)
  ac = tf.concat([attn_l, attn_c], axis=-1)

  # location based attention
  bd = tf.einsum("bcxnh,bynh->bncxy", q_head + r_r_bias, k_head_r) * scale
  bd = rel_shift(bd, row_dim=3, klen=tf.shape(ac)[-1])

  # attention probability
  attn_score = ac + bd
  if attn_mask is not None:
    tf.logging.info("Attention mask shape: %s", attn_mask.shape)
    attn_score = attn_score - INF * attn_mask

  attn_prob = safe_softmax(attn_score, axis=-1)
  monitor_dict["attn_prob"] = attn_prob
  attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)
  attn_prob_l, attn_prob_c = tf.split(
      attn_prob,
      num_or_size_splits=2,
      axis=-1)
  pad_v_head_h = tf.pad(
      v_head_h,
      tf.constant([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]])
  )
  attn_vec = (tf.einsum("bncxy,bcynh->bcxnh",
                        attn_prob_l, pad_v_head_h[:, :-1]) +
              tf.einsum("bncxy,bcynh->bcxnh",
                        attn_prob_c, v_head_h))

  return attn_vec, monitor_dict


def multihead_attn(q, k, v, attn_mask, d_model, n_head, d_head, dropout,
                   dropatt, is_training, kernel_initializer, residual=True,
                   local_attn=False, scope="abs_attn", reuse=None):
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
    if local_attn:
      attn_vec, attn_core_dict = local_abs_attn_core(
          q_head, k_head, v_head, attn_mask, dropatt, is_training, scale)
    else:
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


def rel_multihead_attn(h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                       attn_mask, mems, d_model, n_head, d_head, dropout,
                       dropatt, is_training, kernel_initializer,
                       local_attn=False, scope="rel_attn", reuse=None):
  """Multi-head attention with relative positional encoding."""

  monitor_dict = {}

  scale = 1 / (d_head ** 0.5)
  inp_h = h
  with tf.variable_scope(scope, reuse=reuse):
    if mems is not None and mems.shape.ndims > 1:
      cat = tf.concat([mems, h], 1)
    else:
      cat = h

    q_head_h = head_projection(
        h, d_model, n_head, d_head, kernel_initializer, "q")
    k_head_h = head_projection(
        cat, d_model, n_head, d_head, kernel_initializer, "k")
    v_head_h = head_projection(
        cat, d_model, n_head, d_head, kernel_initializer, "v")

  if FLAGS.share_w:
    with tf.variable_scope(scope, reuse=True):
      # positional heads
      k_head_r = head_projection(
          r, d_model, n_head, d_head, kernel_initializer, "k")
  else:
    with tf.variable_scope(scope, reuse=reuse):
      k_head_r = head_projection(
          r, d_model, n_head, d_head, kernel_initializer, "r")

  with tf.variable_scope(scope, reuse=reuse):
    # core attention ops
    if local_attn:
      attn_vec, attn_core_dict = local_rel_attn_core(
          q_head_h, k_head_h, v_head_h, k_head_r, r_w_bias, r_r_bias,
          attn_mask, dropatt, is_training, scale)
    else:
      attn_vec, attn_core_dict = rel_attn_core(
          q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
          r_r_bias, r_s_bias, attn_mask, dropatt, is_training, scale)

    # post processing
    output, post_dict = post_attention(
        inp_h, attn_vec, d_model, n_head, d_head, dropout,
        is_training, kernel_initializer)

    # things to monitor
    monitor_dict = update_monitor_dict(monitor_dict, attn_core_dict)
    monitor_dict = update_monitor_dict(monitor_dict, post_dict)

  return output, monitor_dict


def rel_position_param(untie_r, n_layer, n_head, d_head, initializer, tf_float):
  """get parameters for relative positional attention."""
  if untie_r:
    r_w_bias = tf.get_variable("r_w_bias", [n_layer, n_head, d_head],
                               dtype=tf_float, initializer=initializer)
    r_r_bias = tf.get_variable("r_r_bias", [n_layer, n_head, d_head],
                               dtype=tf_float, initializer=initializer)
  else:
    r_w_bias = tf.get_variable("r_w_bias", [n_head, d_head],
                               dtype=tf_float, initializer=initializer)
    r_r_bias = tf.get_variable("r_r_bias", [n_head, d_head],
                               dtype=tf_float, initializer=initializer)

  return r_w_bias, r_r_bias


def cache_memory(curr_out, prev_mem, mem_len, reuse_len=None):
  """cache hidden states into memory."""
  if mem_len is None or mem_len == 0:
    return None
  else:
    if reuse_len is not None and reuse_len > 0:
      curr_out = curr_out[:, :reuse_len]

    if prev_mem is None:
      new_mem = curr_out[:, -mem_len:]
    else:
      new_mem = tf.concat([prev_mem, curr_out], 1)[:, -mem_len:]

  return tf.stop_gradient(new_mem)

