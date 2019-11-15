"""Create model function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
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
    get_loss, extract_hiddens, bf16_decorator
except ImportError:
  import model
  from common_ops import causal_attn_mask
  from model_utils import _get_initializer, _get_inp_func, _get_tfm_func, \
    get_loss, extract_hiddens, bf16_decorator

FLAGS = flags.FLAGS


@bf16_decorator
def mlm_loss(features, labels, mems, n_token, is_training):
  """Standard MLM loss as in BERT."""
  del labels
  del mems

  initializer = _get_initializer()

  #### Unpack input
  masked_inp = features["masked_input"]
  type_id = features["type_id"]
  # pos_seq = features["pos_seq"]

  target_mapping = features["target_mapping"]
  target_mask = features["target_mask"]
  target = features["target"]

  monitor_dict = {}

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    masked_embed, word_embed_table = inp_func(
        inputs=masked_inp, type_id=type_id,  # pos_seq=pos_seq,
        return_embed_table=True)

    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    output, _ = tfm_func(
        inputs=masked_embed,
        input_mask=None,
        perm_mask=None)

    lm_loss, _ = model.lm_loss(
        hidden=output,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        target_mapping=None,
        hidden_mapping=target_mapping,
        return_logits=False,
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)

    if FLAGS.sample_strategy == "token_span":
      if target_mask.dtype != tf.float32:
        target_mask = tf.cast(target_mask, tf.float32)
      total_loss = (tf.reduce_sum(lm_loss * target_mask) /
                    tf.reduce_sum(target_mask))
    else:
      total_loss = tf.reduce_mean(lm_loss)

    monitor_dict["lm_loss"] = total_loss

  return total_loss, {}, monitor_dict


@bf16_decorator
def get_lm_loss(features, mems, n_token, is_training):
  """LM loss."""
  del mems

  initializer = _get_initializer()

  #### Unpack input
  inputs = features["inputs"]
  target = features["target"]
  type_id = features["type_id"]
  # pos_seq = features["pos_seq"]

  monitor_dict = {}

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    input_embed, word_embed_table = inp_func(
        inputs=inputs, type_id=type_id,  # pos_seq=pos_seq,
        return_embed_table=True)

    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    output, _ = tfm_func(
        inputs=input_embed,
        input_mask=None,
        perm_mask=None,
        causal=True)

    lm_loss, _ = model.lm_loss(
        hidden=output,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        return_logits=False,
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)

    total_loss = tf.reduce_mean(lm_loss)

    monitor_dict["lm_loss"] = total_loss

  return total_loss, {}, monitor_dict


@bf16_decorator
def electra_loss(features, labels, mems, n_token, is_training):
  """DAE loss with generator."""
  del mems
  del labels
  
  assert FLAGS.ins_ratio + FLAGS.del_ratio == 0.0

  initializer = _get_initializer()
  monitor_dict = {}

  ##### unpack input
  gen_inp = features["gen_inp"]
  gen_tgt = features["gen_tgt"]
  gen_mask_map = features["gen_mask_map"]
  gen_tgt_mask = features["gen_tgt_mask"]

  enc_mask = features["enc_mask"]
  enc_type = features["enc_type"]
  edit_label = features["enc_edit_label"]

  initializer = _get_initializer()

  if FLAGS.use_bfloat16:
    tf_float = tf.bfloat16
  else:
    tf_float = tf.float32

  #### Shared input embedding (for generator)
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    gen_embed, shared_embed_table = inp_func(
        inputs=gen_inp, type_id=enc_type, return_embed_table=True)

  #### Generator TFM
  with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
    gen_tfm_func = _get_tfm_func(initializer,
                                 is_training,
                                 "pretrain",
                                 shrink=FLAGS.gen_shrink)
    gen_output, _ = gen_tfm_func(
        inputs=gen_embed,
        input_mask=enc_mask,
        perm_mask=None)

    gen_lm_loss, _, logits = model.lm_loss(
        hidden=gen_output,
        target=gen_tgt,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=shared_embed_table,
        tie_weight=FLAGS.tie_weight,
        target_mapping=None,
        hidden_mapping=gen_mask_map,
        return_logits=True,
        use_tpu=FLAGS.use_tpu)

    if gen_lm_loss.dtype != tf.float32:
      gen_lm_loss = tf.cast(gen_lm_loss, tf.float32)
    gen_tgt_mask = tf.cast(gen_tgt_mask, gen_lm_loss.dtype)
    gen_loss = (tf.reduce_sum(gen_lm_loss * gen_tgt_mask) /
                tf.reduce_sum(gen_tgt_mask))
    monitor_dict["gen_loss"] = gen_loss

    total_loss = gen_loss

  #### Sample from generator
  uniform = tf.random.uniform(minval=0, maxval=1, shape=logits.shape,
                              dtype=tf.float32)
  gumbel = -tf.log(-tf.log(uniform + 1e-9) + 1e-9)
  samples = tf.argmax(logits + gumbel, -1)
  gen_tokens = tf.cast(samples, tf.int32)

  # map `num_predict` samples to full length
  samples = tf.einsum("bm,bml->bl",
                      tf.cast(samples, tf.float32),
                      tf.cast(gen_mask_map, tf.float32))
  samples = tf.cast(samples, tf.int32)
  enc_inp = tf.where(tf.equal(gen_inp, FLAGS.mask_id), samples, gen_inp)

  #### get the mask for generated same token as the target
  same_mask = tf.equal(gen_tokens, gen_tgt)
  same_mask = tf.cast(same_mask, gen_tgt_mask.dtype) * gen_tgt_mask
  
  # monitor how many generated tokens are the same as the real ones
  same_prec = (tf.reduce_sum(same_mask)
               / tf.reduce_sum(gen_tgt_mask))
  monitor_dict["same_percent"] = same_prec
  
  # If same, change the edit_label to original (0)
  same_mask_full = tf.einsum("bi,bil->bl", same_mask, gen_mask_map)
  edit_label = tf.where(
      tf.cast(same_mask_full, tf.bool),
      tf.zeros(tf.shape(same_mask_full), dtype=edit_label.dtype),
      edit_label)

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_embed = inp_func(inputs=enc_inp, type_id=enc_type)

    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    output, _ = tfm_func(
        inputs=inp_embed,
        input_mask=None,
        perm_mask=None)

    #### edit loss
    edit_loss, edit_logits = model.cls_loss(
        hidden=output,
        target=edit_label,
        n_cls=4,
        d_model=FLAGS.d_model,
        initializer=initializer,
        target_mapping=None,
        hidden_mapping=None,
        return_logits=True,
        scope="edit_type_loss")

    tgt_mask = tf.cast(1.0 - enc_mask, tf.float32)
    edit_loss = (tf.reduce_sum(edit_loss * tgt_mask) /
                 tf.reduce_sum(tgt_mask))

    edit_pred = tf.cast(tf.argmax(edit_logits, axis=-1), dtype=edit_label.dtype)
    edit_corr = tf.cast(tf.equal(edit_pred, edit_label), dtype=tf.float32)
    edit_accu = (tf.reduce_sum(edit_corr * tgt_mask) /
                 tf.reduce_sum(tgt_mask))

    # monitor
    monitor_dict["edit_loss"] = edit_loss
    monitor_dict["accu_edit"] = edit_accu

    def get_class_acc(label_id):
      mask = tf.ones(tf.shape(edit_label), dtype=edit_label.dtype) * label_id
      mask = tf.equal(edit_label, mask)
      mask = tf.cast(mask, dtype=tgt_mask.dtype) * tgt_mask
      accu = tf.reduce_sum(edit_corr * mask) / tf.reduce_sum(mask)
      return accu

    monitor_dict["accu_orig"] = get_class_acc(0)
    if FLAGS.del_ratio > 0:
      monitor_dict["accu_del"] = get_class_acc(FLAGS.del_label)

    # accumulate total loss
    total_loss = FLAGS.edit_weight * edit_loss

    monitor_dict["edit_loss"] = total_loss
    
  return total_loss, {}, monitor_dict
