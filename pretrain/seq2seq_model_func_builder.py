"""Create model function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  import google3.experimental.users.zihangd.pretrain.model as model
  from google3.experimental.users.zihangd.pretrain.common_ops import causal_attn_mask
  from google3.experimental.users.zihangd.pretrain.model_utils import \
      _get_initializer, _get_inp_func, _get_tfm_func, \
      bf16_decorator
except ImportError:
  import model
  from common_ops import causal_attn_mask
  from model_utils import _get_initializer, _get_inp_func, _get_tfm_func, \
      bf16_decorator


FLAGS = flags.FLAGS


@bf16_decorator
def joint_loss(features, labels, n_token, is_training):
  """Decoder only seq2seq."""
  del labels

  initializer = _get_initializer()

  #### Unpack input
  source = features["source"]
  source_pos = features["source_position"]
  source_seg = features["source_segmentation"]

  target = features["target"]
  target_pos = features["target_position"]
  target_seg = features["target_segmentation"]

  # shapes
  bsz = tf.shape(source)[0]
  src_len = tf.shape(source)[1]
  tgt_len = tf.shape(target)[1]

  if FLAGS.use_bfloat16:
    tf_float = tf.bfloat16
  else:
    tf_float = tf.float32

    ##### format inputs
  inputs = tf.concat([source, target], axis=1)
  position = tf.concat([source_pos, target_pos], axis=1)

  src_type_id = features.get(
      "source_type",
      tf.zeros([bsz, src_len], dtype=inputs.dtype))
  tgt_type_id = features.get(
      "target_type",
      tf.ones([bsz, tgt_len], dtype=inputs.dtype))

  if FLAGS.double_type:
    tgt_type_id = tgt_type_id + FLAGS.n_token
  type_id = tf.concat([src_type_id, tgt_type_id], axis=1)

  ##### attention mask: note that `1` indicates CANNOT attend
  # src mask
  src_to_src = tf.not_equal(
      source_seg[:, :, None],
      source_seg[:, None, :])
  src_to_tgt = tf.ones(
      [bsz, src_len, tgt_len],
      dtype=src_to_src.dtype)
  src_mask = tf.concat([src_to_src, src_to_tgt], axis=2)

  # tgt mask
  tgt_to_src = tf.not_equal(
      target_seg[:, :, None],
      source_seg[:, None, :])
  tgt_to_tgt = tf.not_equal(
      target_seg[:, :, None],
      target_seg[:, None, :])
  causal_mask = tf.cast(causal_attn_mask(qlen=tgt_len), tgt_to_tgt.dtype)
  # If any one of them is `1` (indicating cannot attend), i.e. `logical_or`,
  # then the model should NOT attend
  tgt_to_tgt = tf.logical_or(
      tgt_to_tgt,
      causal_mask) 
  tgt_mask = tf.concat([tgt_to_src, tgt_to_tgt], axis=2)

  # concat
  perm_mask = tf.concat([src_mask, tgt_mask], axis=1)
  perm_mask = tf.cast(perm_mask, tf_float)

  # padding
  non_pad_mask = tf.not_equal(target_seg, 0)
  all_eos = tf.constant(FLAGS.eos_id, shape=target.shape, dtype=target.dtype)
  # Replace all <pad> (/P) with <eos> (/S)
  #   - target : /S a1 a2 a3 /S b1 b2 /S c1 c2 /P /P
  #   - tmptgt : /S a1 a2 a3 /S b1 b2 /S c1 c2 /S /S
  tmptgt = tf.where(non_pad_mask, target, all_eos)
  # Shift the `tmptgt` to form the (next-step) prediction target
  #   - target   : \S a1 a2 a3 \S b1 b2 \S c1 c2 \P \P
  #   - pred_tgt : a1 a2 a3 \S b1 b2 \S c1 c2 \S \S \S
  pred_tgt = tf.concat([tmptgt[:, 1:], tmptgt[:, :1]], axis=1)
  loss_mask = tf.cast(non_pad_mask, tf.float32)

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    input_embed, word_embed_table = inp_func(
        inputs=inputs, type_id=type_id, pos_seq=position,
        return_embed_table=True)

    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    output, _ = tfm_func(
        inputs=input_embed,
        input_mask=None,
        perm_mask=perm_mask)

    #### Only predict the target part
    tgt_out = output[:, src_len:]
    tf.logging.info("Output: %s, target output: %s",
                    output.shape, tgt_out.shape)

    lm_loss, nll_loss = model.lm_loss(
        hidden=tgt_out,
        target=pred_tgt,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        label_smooth=FLAGS.label_smooth,
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)

    num_loss = tf.reduce_sum(loss_mask)
    total_loss = tf.reduce_sum(lm_loss * loss_mask) / num_loss
    nll = tf.reduce_sum(nll_loss * loss_mask) / num_loss

    # To be compatible with fairseq, convert to base 2 for logging
  monitor_dict = {
      "loss": total_loss / math.log(2),
      "nll": nll / math.log(2),
      "num_loss": num_loss,
  }

  return total_loss, monitor_dict


@bf16_decorator
def encdec_loss(features, labels, n_token, is_training):
  """Encoder-decoder only seq2seq."""
  del labels

  initializer = _get_initializer()

  #### Unpack input
  source = features["source"]
  source_pos = features["source_position"]
  source_seg = features["source_segmentation"]

  target = features["target"]
  target_pos = features["target_position"]
  target_seg = features["target_segmentation"]

  # shapes
  bsz = tf.shape(source)[0]
  src_len = tf.shape(source)[1]
  tgt_len = tf.shape(target)[1]

  if FLAGS.use_bfloat16:
    tf_float = tf.bfloat16
  else:
    tf_float = tf.float32

  source_type = features.get(
      "source_type",
      tf.zeros([bsz, src_len], dtype=source.dtype))
  target_type = features.get(
      "target_type",
      tf.ones([bsz, tgt_len], dtype=target.dtype))

  ##### attention mask: note that `1` indicates CANNOT attend
  # src-src mask
  src_to_src = tf.cast(
      tf.not_equal(source_seg[:, :, None], source_seg[:, None, :]),
      tf_float)
  # tgt-src mask
  tgt_to_src = tf.cast(
      tf.not_equal(target_seg[:, :, None], source_seg[:, None, :]),
      tf_float)
  # tgt-tgt mask
  tgt_to_tgt = tf.not_equal(
      target_seg[:, :, None],
      target_seg[:, None, :])
  causal_mask = tf.cast(causal_attn_mask(qlen=tgt_len), tgt_to_tgt.dtype)
  # If any one of them is `1` (indicating cannot attend), i.e. `logical_or`,
  # then the model should NOT attend
  tgt_to_tgt = tf.logical_or(
      tgt_to_tgt,
      causal_mask)
  tgt_to_tgt = tf.cast(tgt_to_tgt, tf_float)

  # padding
  non_pad_mask = tf.not_equal(target_seg, 0)
  all_eos = tf.constant(FLAGS.eos_id, shape=target.shape, dtype=target.dtype)
  # Replace all <pad> (/P) with <eos> (/S)
  #   - target : /S a1 a2 a3 /S b1 b2 /S c1 c2 /P /P
  #   - tmptgt : /S a1 a2 a3 /S b1 b2 /S c1 c2 /S /S
  tmptgt = tf.where(non_pad_mask, target, all_eos)
  # Shift the `tmptgt` to form the (next-step) prediction target
  #   - target   : \S a1 a2 a3 \S b1 b2 \S c1 c2 \P \P
  #   - pred_tgt : a1 a2 a3 \S b1 b2 \S c1 c2 \S \S \S
  pred_tgt = tf.concat([tmptgt[:, 1:], tmptgt[:, :1]], axis=1)
  loss_mask = tf.cast(non_pad_mask, tf.float32)

  #### Shared embedding (encoder)
  with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    source_embed, shared_embed_table = inp_func(
        inputs=source, type_id=source_type, pos_seq=source_pos,
        return_embed_table=True)

  #### Encoder
  with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
    enc_tfm_func = _get_tfm_func(initializer,
                                 is_training,
                                 phase="pretrain")
    src_output, _ = enc_tfm_func(
        inputs=source_embed,
        input_mask=None,
        perm_mask=src_to_src)

  #### Shared embedding (decoder)
  with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    target_embed = inp_func(
        inputs=target, type_id=target_type, pos_seq=target_pos)

  #### Decoder
  with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    dec_tfm_func = _get_tfm_func(initializer,
                                 is_training,
                                 phase="pretrain")
    tgt_output, _ = dec_tfm_func(
        inputs=target_embed,
        input_mask=None,
        perm_mask=tgt_to_tgt,
        context=src_output,
        context_mask=tgt_to_src[:, None, :, :])

    lm_loss, nll_loss = model.lm_loss(
        hidden=tgt_output,
        target=pred_tgt,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=shared_embed_table,
        tie_weight=FLAGS.tie_weight,
        label_smooth=FLAGS.label_smooth,
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)

    num_loss = tf.reduce_sum(loss_mask)
    total_loss = tf.reduce_sum(lm_loss * loss_mask) / num_loss
    nll = tf.reduce_sum(nll_loss * loss_mask) / num_loss

  # To be compatible with fairseq, convert to base 2 for logging
  monitor_dict = {
      "loss": total_loss / math.log(2),
      "nll": nll / math.log(2),
      "num_loss": num_loss,
  }

  return total_loss, monitor_dict


@bf16_decorator
def joint_rel_attn_loss(features, labels, n_token, is_training):
  """Decoder only seq2seq."""
  del labels

  initializer = _get_initializer()

  if FLAGS.use_bfloat16:
    tf_float = tf.bfloat16
  else:
    tf_float = tf.float32

  #### Unpack input
  inputs = features["inputs"]
  targets = features["targets"]
  segment = features["segmentation"]
  position = features["position"]
  loss_mask = features["loss_mask"]

  # shapes
  bsz = tf.shape(inputs)[0]
  seq_len = tf.shape(inputs)[1]

  # target type is 1 and source type is 0
  type_id = tf.range(seq_len)
  type_id = tf.math.floormod(type_id, 2)
  target_mask = tf.cast(type_id, tf.bool)
  source_mask = tf.logical_not(target_mask)

  ##### attention mask: note that `1` indicates CANNOT attend
  source_seg = segment
  target_seg = segment
  # src mask
  src_to_src = tf.logical_and(
      tf.not_equal(source_seg[:, :, None], source_seg[:, None, :]),
      tf.logical_and(source_mask[None, :, None], source_mask[None, None, :]))

  src_to_tgt = tf.logical_and(
      tf.ones([bsz, seq_len, seq_len], dtype=src_to_src.dtype),
      tf.logical_and(source_mask[None, :, None], target_mask[None, None, :]))

  # tgt mask
  tgt_to_src = tf.logical_and(
      tf.not_equal(target_seg[:, :, None], source_seg[:, None, :]),
      tf.logical_and(target_mask[None, :, None], source_mask[None, None, :]))

  tgt_to_tgt = tf.not_equal(
      target_seg[:, :, None],
      target_seg[:, None, :])
  causal_mask = tf.cast(causal_attn_mask(qlen=seq_len), tgt_to_tgt.dtype)
  # If any one of them is `1` (indicating cannot attend), i.e. `logical_or`,
  # then the model should NOT attend
  tgt_to_tgt = tf.logical_and(
      tf.logical_or(tgt_to_tgt, causal_mask),
      tf.logical_and(target_mask[None, :, None], target_mask[None, None, :]))
  
  perm_mask = tf.logical_or(
      tf.logical_or(src_to_src, src_to_tgt),
      tf.logical_or(tgt_to_src, tgt_to_tgt))
  perm_mask = tf.cast(perm_mask, tf_float)
  
  target_map = tf.one_hot(tf.range(1, seq_len, delta=2), seq_len)

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    input_embed, word_embed_table = inp_func(
        inputs=inputs, type_id=type_id, pos_seq=position,
        return_embed_table=True)

    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    output, _ = tfm_func(
        inputs=input_embed,
        input_mask=None,
        perm_mask=perm_mask)

    #### Only predict the target part
    tf.logging.info("Output: %s, target output: %s",
                    output.shape, targets.shape)

    lm_loss, nll_loss = model.lm_loss(
        hidden=output,
        target=targets,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        target_mapping=None,
        hidden_mapping=target_map[None,:, :],
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)

    num_loss = tf.reduce_sum(loss_mask)
    total_loss = tf.reduce_sum(lm_loss * loss_mask) / num_loss
    nll = tf.reduce_sum(nll_loss * loss_mask) / num_loss

    # To be compatible with fairseq, convert to base 2 for logging
  monitor_dict = {
      "loss": total_loss / math.log(2),
      "nll": nll / math.log(2),
      "num_loss": num_loss,
  }

  return total_loss, monitor_dict