"""Pretraining TFM model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain.common_ops import update_monitor_dict
  from google3.experimental.users.zihangd.pretrain.common_ops import sinusoid_positional_encoding
  from google3.experimental.users.zihangd.pretrain.common_ops import causal_attn_mask
  from google3.experimental.users.zihangd.pretrain.common_ops import merge_attn_masks
  from google3.experimental.users.zihangd.pretrain.common_ops import embedding_lookup
  from google3.experimental.users.zihangd.pretrain.common_ops import get_activation
  from google3.experimental.users.zihangd.pretrain.common_ops import positionwise_ffn
  from google3.experimental.users.zihangd.pretrain.common_ops import multihead_attn
  from google3.experimental.users.zihangd.pretrain.common_ops import layer_norm
  from google3.experimental.users.zihangd.pretrain.common_ops import rel_encoding
  from google3.experimental.users.zihangd.pretrain.common_ops import consecutive_rel_encoding
except ImportError:
  from common_ops import update_monitor_dict
  from common_ops import sinusoid_positional_encoding
  from common_ops import causal_attn_mask
  from common_ops import merge_attn_masks
  from common_ops import embedding_lookup
  from common_ops import get_activation
  from common_ops import positionwise_ffn
  from common_ops import multihead_attn
  from common_ops import layer_norm
  from common_ops import rel_encoding
  from common_ops import consecutive_rel_encoding
# pylint: enable=g-import-not-at-top


##### Model configuration
# Type
flags.DEFINE_enum("input_proc", "inv_sqrt", ["inv_sqrt", "layer_norm"],
                  help="How to preprocess input.")
flags.DEFINE_bool("rel_attn", False,
                  help="Whether to use relative pattention.")
flags.DEFINE_bool("lm_proj", False,
                  help="Whether to use another projection before the LM loss.")
flags.DEFINE_bool("softmax_bias", False,
                  help="Whether to add softmax bias for the distribution.")

# Size
flags.DEFINE_integer("n_layer", default=6,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
                     help="Dimension of inner hidden size in FFN.")
flags.DEFINE_integer("n_type", default=2,
                     help="Number of type embedding.")
flags.DEFINE_integer("max_pos_len", default=512,
                     help="max length of positional embedding.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

# Hidden non-linearity
flags.DEFINE_string("ff_activation", default="relu",
                    help="Activation type used in position-wise feed-forward.")

# Dropouts
flags.DEFINE_float("dropout", default=0.1,
                   help="Model dropout.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout.")
flags.DEFINE_float("dropact", default=0.0,
                   help="Activation dropout.")

# Summarization layer
flags.DEFINE_string("summary_type", default="first",
                    help="Method used to summarize a sequence into a vector.")

# TFM-XL specific
flags.DEFINE_integer("mem_len", default=0,
                     help="Number of steps to cache")

# Generator specific
flags.DEFINE_integer("gen_shrink", default=3,
                     help="Shrink the hidden dimension of the generator.")

# joint model pretrain specific
flags.DEFINE_bool("double_type", default=False,
                  help="Use type id to differeniate the encoder and decoder "
                  "in the joint model. [0,n_type) for encoder, "
                  "(n_type, 2*n_type] for decoder")

##### Parameter initialization
flags.DEFINE_enum("init", default="truncated_normal",
                  enum_values=["normal", "uniform", "truncated_normal"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def mt_input(inputs, type_id, n_token, n_type, d_embed, dropout, initializer,
             is_training, word_embed_table=None, type_embed_table=None,
             rel_attn=False, pos_seq=None, clamp_len=-1, use_tpu=True,
             return_embed_table=False, reuse=False, scope="input"):
  """Input layer for TFM used in machine translation."""
  tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
  tf.logging.info("===== Input layer =====")
  tf.logging.info("Input related:")
  tf.logging.info("  - inputs %s", inputs)
  tf.logging.info("  - type_id %s", type_id)
  tf.logging.info("  - pos_seq %s", pos_seq)
  tf.logging.info("Hparam related:")
  tf.logging.info("  - n_type %s", n_type)
  tf.logging.info("  - n_token %s", n_token)
  tf.logging.info("============================")

  # scale up the embedding magnitude
  scale = np.sqrt(d_embed)

  with tf.variable_scope(scope, reuse=reuse):
    ##### Shape
    seq_len = tf.shape(inputs)[1]

    ##### Word embedding
    word_emb, word_embed_table = embedding_lookup(
        x=inputs,
        n_token=n_token,
        d_embed=d_embed,
        initializer=initializer,
        lookup_table=word_embed_table,
        use_tpu=use_tpu,
        dtype=tf_float,
        scope="word_embedding")
    word_emb = word_emb * scale

    ##### Type embedding
    if type_id is not None:
      type_emb, type_embed_table = embedding_lookup(
          x=type_id,
          n_token=n_type,
          d_embed=d_embed,
          initializer=initializer,
          lookup_table=type_embed_table,
          use_tpu=use_tpu,
          dtype=tf_float,
          scope="type_embedding")
      word_emb += type_emb * scale

    ##### Absolute positional embedding
    if not rel_attn:
      with tf.variable_scope("pos_embedding", reuse=reuse):
        if pos_seq is None:
          pos_seq = tf.range(0, seq_len, 1.0)
        pos_emb = sinusoid_positional_encoding(
            pos_seq, d_embed, clamp_len=clamp_len, dtype=tf_float)
        word_emb += pos_emb

    ##### Dropout
    output = tf.layers.dropout(word_emb, dropout, training=is_training)

  if return_embed_table:
    return output, word_embed_table
  else:
    return output


def input_layer(inputs, type_id, n_token, n_type, n_pos, d_embed, dropout,
                initializer, is_training, word_embed_table=None,
                type_embed_table=None, pos_embed_table=None, use_tpu=True,
                rel_attn=False, return_embed_table=False, reuse=False,
                scope="input"):
  """Turn inputs tokens to embedding."""

  tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
  with tf.variable_scope(scope, reuse=reuse):
    ##### Shape
    seq_len = tf.shape(inputs)[1]

    ##### Word embedding
    word_emb, word_embed_table = embedding_lookup(
        x=inputs,
        n_token=n_token,
        d_embed=d_embed,
        initializer=initializer,
        lookup_table=word_embed_table,
        use_tpu=use_tpu,
        dtype=tf_float,
        scope="word_embedding")

    ##### Segment embedding
    if type_id is not None:
      type_emb, type_embed_table = embedding_lookup(
          x=type_id,
          n_token=n_type,
          d_embed=d_embed,
          initializer=initializer,
          lookup_table=type_embed_table,
          use_tpu=use_tpu,
          dtype=tf_float,
          scope="segment_embedding")
      word_emb += type_emb
    else:
      type_embed_table = None

    ##### Absolute positional embedding
    if not rel_attn:
      with tf.variable_scope("pos_embedding", reuse=reuse):
        if pos_embed_table is None:
          pos_embed_table = tf.get_variable("lookup_table", [n_pos, d_embed],
                                            initializer=initializer,
                                            dtype=tf_float)
        pos_emb = pos_embed_table[:seq_len]
        pos_emb = tf.broadcast_to(pos_emb[None], tf.shape(word_emb))
        word_emb += pos_emb

    ##### Input embedding layer normalization and dropout
    word_emb = layer_norm(
        word_emb, begin_norm_axis=-1, reuse=reuse, scope="LayerNorm")
    output = tf.layers.dropout(word_emb, dropout, training=is_training)

  if return_embed_table:
    return output, word_embed_table
  else:
    return output


def transformer(inputs, n_layer, d_model, n_head, d_head, d_inner,
                dropout, dropatt, dropact, initializer, is_training,
                context=None, context_mask=None, input_mask=None,
                perm_mask=None, ff_activation="relu", causal=False,
                rel_attn=False, pos_seq=None, clamp_len=-1,
                return_all_hidden=False, scope="transformer"):
  """Transformer model."""

  monitor_dict = {}
  if input_mask is not None:
    monitor_dict["inp_mask"] = input_mask

  tf.logging.info("===== Transformer =====")
  tf.logging.info("Input related:")
  tf.logging.info("  - inputs %s", inputs)
  tf.logging.info("  - input_mask %s", input_mask)
  tf.logging.info("  - perm_mask %s", perm_mask)
  tf.logging.info("  - context %s", context)
  tf.logging.info("  - context_mask %s", context_mask)
  tf.logging.info("Hparam related:")
  tf.logging.info("  - initializer %s", initializer)
  tf.logging.info("  - ff_activation %s", ff_activation)
  tf.logging.info("  - causal %s", causal)
  tf.logging.info("============================")

  hiddens = []
  with tf.variable_scope(scope):
    ##### Attention mask
    if causal:
      causal_mask = causal_attn_mask(tf.shape(inputs)[1], dtype=inputs.dtype)
      causal_mask = causal_mask[None, None]
    else:
      causal_mask = None
    attn_mask = merge_attn_masks(causal_mask, input_mask, perm_mask)

    ##### Input projection
    if inputs.shape.as_list()[-1] != d_model:
      tf.logging.info("Project input embedding: %d -> %d",
                      inputs.shape.as_list()[-1], d_model)
      output = tf.layers.dense(inputs, d_model, activation=None,
                               kernel_initializer=initializer,
                               name="input_projection")
    else:
      output = inputs

    hiddens.append(output)

    ##### Get relative attention bias
    if rel_attn:
      tf.logging.info("Use relative attention")
      if pos_seq is None:
        seq_len = tf.shape(output)[1]
        attn_bias = consecutive_rel_encoding(
            seq_len, d_model, n_head, clamp_len, dropout, is_training,
            initializer, output.dtype)
      else:
        attn_bias = rel_encoding(
            pos_seq, pos_seq, d_model, n_head, clamp_len, dropout, is_training,
            initializer, output.dtype)
    else:
      attn_bias = None

    ##### Attention layers
    for i in range(n_layer):
      with tf.variable_scope("layer_{}".format(i)):
        output, attn_dict = multihead_attn(
            q=output,
            k=output,
            v=output,
            attn_mask=attn_mask,
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer,
            attn_bias=attn_bias,
            scope="self_attn")

        if context is not None:
          output, _ = multihead_attn(
              q=output,
              k=context,
              v=context,
              attn_mask=context_mask,
              d_model=d_model,
              n_head=n_head,
              d_head=d_head,
              dropout=dropout,
              dropatt=dropatt,
              is_training=is_training,
              kernel_initializer=initializer,
              scope="cross_attn")

        output, ffn_dict = positionwise_ffn(
            inp=output,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            dropact=dropact,
            initializer=initializer,
            activation_type=ff_activation,
            is_training=is_training)

        hiddens.append(output)

        # Update monitor dict
        monitor_dict = update_monitor_dict(
            monitor_dict, attn_dict, prefix="layer_{}_attn".format(i))
        monitor_dict = update_monitor_dict(
            monitor_dict, ffn_dict, prefix="layer_{}_ff".format(i))

    if return_all_hidden:
      return hiddens, monitor_dict
    else:
      return output, monitor_dict


def xlnet(
    input_k, input_cat, mapping, n_layer, d_model, n_head, d_head, d_inner,
    dropout, dropatt, dropact, initializer, is_training, input_mask=None,
    perm_mask=None, ff_activation="gelu", clamp_len=-1,
    scope="transformer"):
  """XLNet."""

  tf.logging.info("===== XLNet =====")
  tf.logging.info("Input related:")
  tf.logging.info("  - input_k %s", input_k)
  tf.logging.info("  - input_cat %s", input_cat)
  tf.logging.info("  - input_mask %s", input_mask)
  tf.logging.info("  - perm_mask %s", perm_mask)
  tf.logging.info("  - mapping %s", mapping)
  tf.logging.info("Hparam related:")
  tf.logging.info("  - initializer %s", initializer)
  tf.logging.info("  - ff_activation %s", ff_activation)
  tf.logging.info("=============================")

  monitor_dict = {}
  with tf.variable_scope(scope):
    ##### Shapes
    bsz = tf.shape(input_k)[0]
    seq_len = tf.shape(input_k)[1]

    ##### Attention mask
    attn_mask = merge_attn_masks(None, input_mask, perm_mask)

    ##### Relative position and segment encoding
    attn_bias_k = consecutive_rel_encoding(
        seq_len, d_model, n_head, clamp_len, dropout, is_training,
        initializer, input_cat.dtype)
    attn_bias_q = tf.einsum("nij,bki->bnkj", attn_bias_k, mapping)
    attn_bias_k = tf.broadcast_to(attn_bias_k[None],
                                  [bsz, n_head, seq_len, seq_len])
    attn_bias = tf.concat([attn_bias_k, attn_bias_q], -2)

    ##### Concat into one sequence
    output = input_cat
    if output.shape.as_list()[-1] != d_model:
      tf.logging.info("Project input embedding: %d -> %d",
                      output.shape.as_list()[-1], d_model)
      output = tf.layers.dense(output, d_model, activation=None,
                               kernel_initializer=initializer,
                               name="input_projection")

    for i in range(n_layer):
      with tf.variable_scope("layer_{}".format(i)):
        output, attn_dict = multihead_attn(
            q=output,
            k=output[:, :seq_len],
            v=output[:, :seq_len],
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer)

        output, ffn_dict = positionwise_ffn(
            inp=output,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            dropact=dropact,
            initializer=initializer,
            activation_type=ff_activation,
            is_training=is_training)

        # Update monitor dict
        monitor_dict = update_monitor_dict(
            monitor_dict, attn_dict, prefix="layer_{}_attn".format(i))
        monitor_dict = update_monitor_dict(
            monitor_dict, ffn_dict, prefix="layer_{}_ff".format(i))

    return output[:, :seq_len], output[:, seq_len:], monitor_dict


def cls_loss(hidden, target, n_cls, d_model, initializer, hidden_mapping=None,
             target_mapping=None, return_logits=False, use_tpu=False,
             use_proj=False, scope="cls_loss", reuse=False):
  """Compute classification cross entropy loss."""

  tf.logging.info("===== Language model loss =====")
  tf.logging.info("  - hidden_mapping %s", hidden_mapping)
  tf.logging.info("  - target_mapping %s", target_mapping)
  tf.logging.info("===============================")

  if hidden_mapping is not None:
    hidden = tf.einsum("bld,bml->bmd", hidden, hidden_mapping)

  if target_mapping is not None:
    target = tf.einsum("bl,bml->bm", target, target_mapping)

  if use_proj or hidden.shape.as_list()[-1] != d_model:
    with tf.variable_scope("cls_proj"):
      hidden = tf.layers.dense(
          hidden,
          units=d_model,
          activation=get_activation(FLAGS.ff_activation),
          kernel_initializer=initializer)

  with tf.variable_scope(scope, reuse=reuse):
    softmax_w = tf.get_variable("weight", [n_cls, d_model],
                                dtype=hidden.dtype, initializer=initializer)
    softmax_b = tf.get_variable("bias", [n_cls], dtype=hidden.dtype,
                                initializer=tf.zeros_initializer())
    logits = tf.einsum("bid,nd->bin", hidden, softmax_w) + softmax_b

    if logits.dtype != tf.float32:
      # Always use float32 for cross entropy loss
      logits = tf.cast(logits, tf.float32)

    if use_tpu:
      onehot_target = tf.one_hot(target, n_cls, dtype=logits.dtype)
      loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * onehot_target, -1)
    else:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                            logits=logits)

    if return_logits:
      return loss, logits
    else:
      return loss


def lm_loss(hidden, target, n_token, d_model, initializer, lookup_table=None,
            tie_weight=False, hidden_mapping=None, target_mapping=None,
            label_smooth=None, return_logits=False, use_tpu=False):
  """Compute language modeling cross entropy loss."""

  tf.logging.info("===== Language model loss =====")
  tf.logging.info("  - tie_weight %s", tie_weight)
  tf.logging.info("  - hidden_mapping %s", hidden_mapping)
  tf.logging.info("  - target_mapping %s", target_mapping)
  tf.logging.info("  - label_smooth %s", label_smooth)
  tf.logging.info("  - lm_proj %s", FLAGS.lm_proj)
  tf.logging.info("===============================")

  if hidden_mapping is not None:
    hidden = tf.einsum("bld,bml->bmd", hidden, hidden_mapping)

  if target_mapping is not None:
    target = tf.cast(
        tf.einsum("bl,bml->bm",
                  tf.cast(target, tf.float32),
                  tf.cast(target_mapping, tf.float32)),
        target.dtype)

  # Apply one more non-linear transformation before the output layer.
  # This matrix is not used after pre-training.
  if FLAGS.lm_proj or hidden.shape.as_list()[-1] != d_model:
    with tf.variable_scope("lm_proj"):
      hidden = tf.layers.dense(
          hidden,
          units=d_model,
          activation=get_activation(FLAGS.ff_activation),
          kernel_initializer=initializer)
      hidden = layer_norm(hidden, begin_norm_axis=-1, scope="LayerNorm")

  with tf.variable_scope("lm_loss"):
    if tie_weight:
      assert lookup_table is not None, \
          "lookup_table cannot be None for tie_weight"
      softmax_w = lookup_table
    else:
      softmax_w = tf.get_variable("weight", [n_token, d_model],
                                  dtype=hidden.dtype, initializer=initializer)

    logits = tf.einsum("bid,nd->bin", hidden, softmax_w)
    if FLAGS.softmax_bias:
      softmax_b = tf.get_variable("bias", [n_token], dtype=hidden.dtype,
                                  initializer=tf.zeros_initializer())
      logits += softmax_b

    if logits.dtype != tf.float32:
      # Always use float32 for LM loss
      logits = tf.cast(logits, tf.float32)

    if use_tpu:
      onehot_target = tf.one_hot(target, n_token, dtype=logits.dtype)
      nll = -tf.reduce_sum(tf.nn.log_softmax(logits) * onehot_target, -1)
      if label_smooth is not None:
        smooth_target = (onehot_target * (1 - label_smooth) +
                         label_smooth / n_token)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * smooth_target, -1)
      else:
        loss = nll
    else:
      nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                           logits=logits)
      if label_smooth is not None:
        onehot_target = tf.one_hot(target, n_token, dtype=logits.dtype)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_target,
                                               logits=logits)
      else:
        loss = nll

    if return_logits:
      return loss, nll, logits
    else:
      return loss, nll


def binary_loss(hidden, target, d_model, initializer, hidden_mapping=None,
                target_mapping=None, reuse=False):
  """Compute language modeling cross entropy loss."""

  tf.logging.info("===== Binary loss =====")
  tf.logging.info("  - hidden_mapping %s", hidden_mapping)
  tf.logging.info("  - target_mapping %s", target_mapping)
  tf.logging.info("===============================")

  if hidden_mapping is not None:
    hidden = tf.einsum("bld,bml->bmd", hidden, hidden_mapping)

  if target_mapping is not None:
    target = tf.einsum("bl,bml->bm", target, target_mapping)

  # Apply one more non-linear transformation before the output layer.
  # This matrix is not used after pre-training.
  with tf.variable_scope("binary_proj", reuse=reuse):
    hidden = tf.layers.dense(
        hidden,
        units=d_model,
        activation=get_activation(FLAGS.ff_activation),
        kernel_initializer=initializer)
    # (zihangd): electra removes this layer norm for unknown reason
    # hidden = layer_norm(hidden, begin_norm_axis=-1, scope="LayerNorm")

  with tf.variable_scope("binary_loss", reuse=reuse):
    binary_w = tf.get_variable("weight", [d_model],
                               dtype=hidden.dtype, initializer=initializer)

    binary_b = tf.get_variable("bias", [1], dtype=hidden.dtype,
                               initializer=tf.zeros_initializer())

    logits = tf.einsum("bid,d->bi", hidden, binary_w) + binary_b
    if logits.dtype != tf.float32:
      # Always use float32 for loss
      logits = tf.cast(logits, tf.float32)

    target = tf.cast(target, dtype=logits.dtype)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=logits)

    # whether predictions are correct
    prediction = tf.cast(logits > 0, target.dtype)
    correct = tf.cast(tf.equal(prediction, target), tf.float32)

  return loss, correct


def summarize_sequence(summary_type, hidden, d_model, n_head, d_head, dropout,
                       dropatt, input_mask, is_training, initializer,
                       scope=None, reuse=None):
  """Summarize hidden sequence into a vector."""

  tf.logging.info("===== Sequence summary =====")
  tf.logging.info("  - input_mask %s", input_mask)
  tf.logging.info("  - summary_type %s", summary_type)
  tf.logging.info("============================")

  with tf.variable_scope(scope, "sequnece_summary", reuse=reuse):
    if summary_type == "last":
      summary = hidden[:, -1]
    elif summary_type == "first":
      summary = hidden[:, 0]
    elif summary_type == "max":
      if input_mask is None:
        summary = tf.reduce_max(hidden, axis=1)
      else:
        neg_pad = -1e10 * input_mask[:, :, None]
        summary = tf.reduce_max(hidden + neg_pad, axis=1)
    elif summary_type == "mean":
      if input_mask is None:
        summary = tf.reduce_mean(hidden, axis=1)
      else:
        inp_mask = (1. - input_mask)[:, :, None]
        summary = (tf.reduce_sum(hidden * inp_mask, axis=1) /
                   (1e-6 + tf.reduce_sum(inp_mask, axis=1)))
    elif summary_type == "attn":
      bsz = tf.shape(hidden)[1]

      summary_bias = tf.get_variable("summary_bias", [d_model],
                                     dtype=hidden.dtype,
                                     initializer=initializer)
      summary_bias = tf.tile(summary_bias[None, None], [bsz, 1, 1])

      if input_mask is not None:
        # [B X T] -> [B x N x F x T]
        input_mask = input_mask[:, None, None, :]

      summary, _ = multihead_attn(summary_bias, hidden, hidden, input_mask,
                                  d_model, n_head, d_head, dropout, dropatt,
                                  is_training, initializer, residual=False)
      summary = summary[:, 0]
    else:
      raise ValueError("Unsupported summary type {}".format(summary_type))

    # use another projection with `tanh` activation
    summary = tf.layers.dense(
        summary,
        d_model,
        activation=tf.tanh,
        use_bias=True,
        kernel_initializer=initializer,
        name="summary")

  return summary

