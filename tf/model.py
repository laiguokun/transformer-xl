"""Pretraining models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.lm.common_ops import absolute_positional_encoding
  from google3.experimental.users.zihangd.lm.common_ops import cache_memory
  from google3.experimental.users.zihangd.lm.common_ops import causal_attn_mask
  from google3.experimental.users.zihangd.lm.common_ops import embedding_lookup
  from google3.experimental.users.zihangd.lm.common_ops import get_activation
  from google3.experimental.users.zihangd.lm.common_ops import layer_norm
  from google3.experimental.users.zihangd.lm.common_ops import local_attn_mask
  from google3.experimental.users.zihangd.lm.common_ops import multihead_attn
  from google3.experimental.users.zihangd.lm.common_ops import positionwise_ffn
  from google3.experimental.users.zihangd.lm.common_ops import rel_multihead_attn
  from google3.experimental.users.zihangd.lm.common_ops import rel_position_param
  from google3.experimental.users.zihangd.lm.common_ops import relative_positional_encoding
  from google3.experimental.users.zihangd.lm.common_ops import update_monitor_dict
except ImportError as e:
  from common_ops import absolute_positional_encoding
  from common_ops import cache_memory
  from common_ops import causal_attn_mask
  from common_ops import embedding_lookup
  from common_ops import get_activation
  from common_ops import layer_norm
  from common_ops import local_attn_mask
  from common_ops import multihead_attn
  from common_ops import positionwise_ffn
  from common_ops import rel_multihead_attn
  from common_ops import rel_position_param
  from common_ops import relative_positional_encoding
  from common_ops import update_monitor_dict
# pylint: enable=g-import-not-at-top


flags.DEFINE_bool("inp_lnorm", default=False,
                  help="Whether to apply layer norm to the input embedding.")
flags.DEFINE_bool("use_lm_proj", default=False,
                  help="Whether to use another projection before computing "
                  "logits.")
flags.DEFINE_bool("shaw_rel_pos", default=False,
                  help="Whether to use Shaw's relative positional attention.")
flags.DEFINE_string("chunk_len", default=None, help="")
flags.DEFINE_string("chunk_layer", default=None, help="")

FLAGS = flags.FLAGS


def tfm(inputs, n_token, n_layer, d_model, n_head, d_head, d_inner,
        dropout, dropatt, attn_type, initializer, is_training, bi_data,
        max_pos_embed=512, input_mask=None, use_tpu=True, ff_activation="gelu",
        mems=None, use_bfloat16=False, scope="tfm"):
  """transformer language model."""

  # never use mems in bert: just to keep the interface the same
  del mems

  monitor_dict = {}
  if input_mask is not None:
    monitor_dict["inp_mask"] = input_mask

  tf_float = tf.bfloat16 if use_bfloat16 else tf.float32
  tf.logging.info("===== TransformerLM =====")
  tf.logging.info("Use float type %s", tf_float)
  tf.logging.info("Input related:")
  tf.logging.info("  - inputs %s", inputs)
  tf.logging.info("  - input_mask %s", input_mask)
  tf.logging.info("Hparam related:")
  tf.logging.info("  - bi_data %s", bi_data)
  tf.logging.info("  - attn_type %s", attn_type)
  tf.logging.info("  - initializer %s", initializer)
  tf.logging.info("  - ff_activation %s", ff_activation)
  tf.logging.info("============================")

  with tf.variable_scope(scope):
    seq_len = tf.shape(inputs)[1]
    bsz = tf.shape(inputs)[0]

    ##### Attention mask
    # causal attention mask
    if attn_type == "uni":
      attn_mask = causal_attn_mask(seq_len, 0, tf_float, False)
      attn_mask = attn_mask[None, None, :, :]
    elif attn_type == "bi":
      attn_mask = None
    else:
      raise ValueError("Unsupported attention type: {}".format(attn_type))

    ##### Word embedding
    word_embed, word_embed_table = embedding_lookup(
        x=inputs,
        n_token=n_token,
        d_embed=d_model,
        initializer=initializer,
        use_tpu=use_tpu,
        dtype=tf_float,
        scope="word_embedding")

    ##### Absolute positional embedding
    pos_embed = absolute_positional_encoding(
        seq_len, d_model, bi_data, bsz=bsz, clamp_len=-1,
        dtype=tf_float, sinusoid=False, max_len=max_pos_embed,
        initializer=initializer)

    word_embed += pos_embed

    ##### Input embedding layer normalization and dropout
    with tf.variable_scope("input"):
      word_embed = layer_norm(
          word_embed, begin_norm_axis=-1, reuse=False, scope="LayerNorm")
      output_h = tf.layers.dropout(word_embed, dropout, training=is_training)

    ##### Prepare local attention
    if FLAGS.chunk_len is not None:
      chunk_len_list = [int(i) for i in FLAGS.chunk_len.split(",")]
      if FLAGS.chunk_layer is not None and FLAGS.chunk_layer:
        chunk_layer_list = [int(i) for i in FLAGS.chunk_layer.split(",")]
        assert len(chunk_len_list) == len(chunk_layer_list)
      else:
        group_size = (n_layer + len(chunk_len_list) - 1) // len(chunk_len_list)
        chunk_layer_list = [group_size * i for i in range(len(chunk_len_list))]

      chunk_mapping = {}
      for chunk_len, chunk_layer in zip(chunk_len_list, chunk_layer_list):
        chunk_mapping[chunk_layer] = chunk_len
      tf.logging.info("Get chunk mapping: %s", chunk_mapping)

    ##### Attention layers
    for i in range(n_layer):
      # Reshape for local attention
      if FLAGS.chunk_len is not None and i in chunk_mapping:
        chunk_len_i = chunk_mapping[i]
        if chunk_len_i == -1:
          tgt_shape = [bsz, seq_len, d_model]
        else:
          tgt_shape = [bsz, seq_len // chunk_len_i, chunk_len_i, d_model]

        attn_mask = local_attn_mask(input_mask, chunk_len_i)
        output_h = tf.reshape(output_h, tgt_shape)

        tf.logging.info("== Change to chunk len %d at layer %d ==",
                        chunk_len_i, i)
        tf.logging.info("  - output_h: %s", output_h.shape)
        if attn_mask is not None:
          tf.logging.info("  - attn_mask: %s", attn_mask.shape)

      with tf.variable_scope("layer_{}".format(i)):
        local_attn = (FLAGS.chunk_len is not None and
                      output_h.shape.ndims == 4)
        tf.logging.info("Layer %d local attention: %s", i, local_attn)
        output_h, attn_dict = multihead_attn(
            q=output_h,
            k=output_h,
            v=output_h,
            attn_mask=attn_mask,
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer,
            local_attn=local_attn,
            reuse=False)

        output_h, ffn_h_dict = positionwise_ffn(
            inp=output_h,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            kernel_initializer=initializer,
            activation_type=ff_activation,
            is_training=is_training,
            reuse=False)

        # Update monitor dict
        monitor_dict = update_monitor_dict(
            monitor_dict, attn_dict, prefix="layer_{}_attn".format(i))
        monitor_dict = update_monitor_dict(
            monitor_dict, ffn_h_dict, prefix="layer_{}_ff".format(i))

    if FLAGS.chunk_len is not None and output_h.shape.ndims == 4:
      output_h = tf.reshape(output_h, [bsz, seq_len, d_model])
      tf.logging.info("Reshape final output_h to %s", output_h.shape)

    return output_h, [], word_embed_table, monitor_dict


def tfm_xl(inputs, mems, n_token, n_layer, d_model, n_head, d_head, d_inner,
           dropout, dropatt, attn_type, bi_data, initializer, is_training,
           mem_len=None, same_length=False, clamp_len=-1, untie_r=False,
           use_tpu=True, input_mask=None, ff_activation="gelu",
           use_bfloat16=False, scope="tfm_xl"):
  """Transform-xl."""

  tf_float = tf.bfloat16 if use_bfloat16 else tf.float32
  tf.logging.info("===== Transformer-XL LM=====")
  tf.logging.info("Use float type %s", tf_float)
  tf.logging.info("Input related:")
  tf.logging.info("  - mems %s", mems)
  tf.logging.info("  - inputs %s", inputs)
  tf.logging.info("  - input_mask %s", input_mask)
  tf.logging.info("Hparam related:")
  tf.logging.info("  - bi_data %s", bi_data)
  tf.logging.info("  - untie_r %s", untie_r)
  tf.logging.info("  - attn_type %s", attn_type)
  tf.logging.info("  - mem_len %s", mem_len)
  tf.logging.info("  - initializer %s", initializer)
  tf.logging.info("  - ff_activation %s", ff_activation)
  tf.logging.info("  - inp_lnorm %s", FLAGS.inp_lnorm)
  tf.logging.info("=============================")

  monitor_dict = {}
  if input_mask is not None:
    monitor_dict["inp_mask"] = input_mask

  new_mems = []
  with tf.variable_scope(scope):
    # parameters for relative position attention
    r_w_bias, r_r_bias = rel_position_param(
        untie_r, n_layer, n_head, d_head, initializer, tf_float)

    # sizes
    bsz = tf.shape(inputs)[0]
    qlen = tf.shape(inputs)[1]
    mlen = tf.shape(mems[0])[1] if mems is not None else 0
    klen = mlen + qlen

    ##### Attention mask
    # causal attention mask
    if attn_type == "uni":
      full_attn_mask = causal_attn_mask(qlen, mlen, tf_float, same_length)
      # [F x T'] -> [B x N x F x T']
      full_attn_mask = full_attn_mask[None, None, :, :]
    elif attn_type == "bi":
      full_attn_mask = None
    else:
      raise ValueError("Unsupported attention type: {}".format(attn_type))

    ##### Word embedding
    word_embed, lookup_table = embedding_lookup(
        x=inputs,
        n_token=n_token,
        d_embed=d_model,
        initializer=initializer,
        use_tpu=use_tpu,
        dtype=tf_float,
        scope="word_embedding")

    # Input layernorm
    if FLAGS.inp_lnorm:
      with tf.variable_scope("input"):
        word_embed = layer_norm(
            word_embed, begin_norm_axis=-1, reuse=False, scope="LayerNorm")

    # Apply dropout to the input of the attention layers
    output_h = tf.layers.dropout(word_embed, dropout, training=is_training)

    ##### Relative positional encoding
    full_rel_pos_emb = relative_positional_encoding(
        qlen, klen, d_model, bi_data, attn_type,
        bsz=bsz, clamp_len=clamp_len, dtype=tf_float)
    full_rel_pos_emb = tf.layers.dropout(
        full_rel_pos_emb, dropout, training=is_training)

    ##### Prepare for local attention
    if FLAGS.chunk_len is not None:
      chunk_len_list = [int(i) for i in FLAGS.chunk_len.split(",")]
      if FLAGS.chunk_layer is not None and FLAGS.chunk_layer:
        chunk_layer_list = [int(i) for i in FLAGS.chunk_layer.split(",")]
        assert len(chunk_len_list) == len(chunk_layer_list)
      else:
        group_size = (n_layer + len(chunk_len_list) - 1) // len(chunk_len_list)
        chunk_layer_list = [group_size * i for i in range(len(chunk_len_list))]

      chunk_mapping = {}
      for chunk_len, chunk_layer in zip(chunk_len_list, chunk_layer_list):
        chunk_mapping[chunk_layer] = chunk_len
      tf.logging.info("Get chunk mapping: %s", chunk_mapping)
    else:
      rel_pos_emb = full_rel_pos_emb
      attn_mask = full_attn_mask

    ##### Attention layers
    if mems is None:
      mems = [None] * n_layer

    for i in range(n_layer):
      # cache new mems
      new_mems.append(cache_memory(output_h, mems[i], mem_len))

      # local attention
      if FLAGS.chunk_len is not None and i in chunk_mapping:
        chunk_len_i = chunk_mapping[i]
        if chunk_len_i == -1:
          tgt_shape = [bsz, qlen, d_model]
          rel_pos_emb = full_rel_pos_emb
          attn_mask = full_attn_mask
        else:
          tgt_shape = [bsz, qlen // chunk_len_i, chunk_len_i, d_model]
          rel_pos_emb = full_rel_pos_emb[:, -2 * chunk_len_i - 1:]
          attn_mask = full_attn_mask[:, :, -chunk_len_i:, -2 * chunk_len_i:]

        output_h = tf.reshape(output_h, tgt_shape)

        tf.logging.info("== Change to chunk len %d at layer %d ==",
                        chunk_len_i, i)
        tf.logging.info("  - output_h: %s", output_h.shape)
        tf.logging.info("  - rel_pos_emb: %s", rel_pos_emb.shape)
        if attn_mask is not None:
          tf.logging.info("  - attn_mask: %s", attn_mask.shape)

      with tf.variable_scope("layer_{}".format(i)):
        local_attn = (FLAGS.chunk_len is not None and
                      output_h.shape.ndims == 4)
        tf.logging.info("Layer %d local attention: %s", i, local_attn)
        output_h, attn_dict = rel_multihead_attn(
            h=output_h,
            r=rel_pos_emb,
            r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
            r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
            seg_mat=None,
            r_s_bias=None,
            seg_embed=None,
            attn_mask=attn_mask,
            mems=mems[i],
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer,
            local_attn=local_attn,
            reuse=False)

        #### feed-forward
        output_h, ffn_h_dict = positionwise_ffn(
            inp=output_h,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            kernel_initializer=initializer,
            activation_type=ff_activation,
            is_training=is_training,
            reuse=False)

        # Update monitor dict with attention dict
        monitor_dict = update_monitor_dict(
            monitor_dict, attn_dict, prefix="layer_{}_attn".format(i))
        monitor_dict = update_monitor_dict(
            monitor_dict, ffn_h_dict, prefix="layer_{}_ff".format(i))

    if FLAGS.chunk_len is not None and output_h.shape.ndims == 4:
      output_h = tf.reshape(output_h, [bsz, qlen, d_model])
      tf.logging.info("Reshape final output_h to %s", output_h.shape)

    return output_h, new_mems, lookup_table, monitor_dict


def lm_loss(hidden, target, n_token, d_model, initializer, lookup_table=None,
            tie_weight=False, hidden_mapping=None, target_mapping=None,
            use_tpu=False):
  """Compute language modeling cross entropy loss."""

  tf.logging.info("===== Language model loss =====")
  tf.logging.info("  - tie_weight %s", tie_weight)
  tf.logging.info("  - use_lm_proj %s", FLAGS.use_lm_proj)
  tf.logging.info("  - hidden_mapping %s", hidden_mapping)
  tf.logging.info("  - target_mapping %s", target_mapping)
  tf.logging.info("===============================")

  if hidden_mapping is not None:
    hidden = tf.einsum("bld,bml->bmd", hidden, hidden_mapping)

  if target_mapping is not None:
    target = tf.einsum("bl,bml->bm", target, target_mapping)

  if FLAGS.use_lm_proj:
    # Apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("lm_proj"):
      hidden = tf.layers.dense(
          hidden,
          units=d_model,
          activation=get_activation("gelu"),
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

    softmax_b = tf.get_variable("bias", [n_token], dtype=hidden.dtype,
                                initializer=tf.zeros_initializer())

    logits = tf.einsum("bid,nd->bin", hidden, softmax_w) + softmax_b
    if logits.dtype != tf.float32:
      logits = tf.cast(logits, tf.float32)

    if use_tpu:
      one_hot_target = tf.one_hot(target, n_token, dtype=logits.dtype)
      loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)
    else:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                            logits=logits)

    return loss

