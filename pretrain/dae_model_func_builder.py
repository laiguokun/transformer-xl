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
def dae_loss(features, labels, mems, n_token, is_training):
  """DAE loss with generator."""
  del mems
  del labels

  initializer = _get_initializer()
  monitor_dict = {}

  ##### unpack input
  gen_inp = features["gen_inp"]
  gen_tgt = features["gen_tgt"]
  gen_mask_map = features["gen_mask_map"]
  gen_tgt_mask = features["gen_tgt_mask"]

  enc_mask = features["enc_mask"]
  enc_type = features["enc_type"]

  dec_inp = features["dec_inp"]
  dec_tgt = features["dec_tgt"]
  dec_mask = features["dec_mask"]
  dec_type = features["dec_type"]
  edit_label = features["edit_label"]
  dec_mask_map = features["dec_mask_map"]
  dec_masked_tgt = features["dec_masked_tgt"]
  dec_lm_tgt_mask = features["dec_lm_tgt_mask"]
  rep_enc2dec_full = features["rep_enc2dec_full"]
  rep_enc2dec_part = features["rep_enc2dec_part"]

  #### Shared input embedding (for generator)
  with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
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
  same_mask = (tf.cast(same_mask, rep_enc2dec_full.dtype) *
               tf.cast(gen_tgt_mask, rep_enc2dec_full.dtype))

  # monitor how many generated tokens are the same as the real ones
  same_prec = (tf.reduce_sum(tf.cast(same_mask, gen_tgt_mask.dtype))
               / tf.reduce_sum(gen_tgt_mask))
  monitor_dict["same_percent"] = same_prec

  # If same, change the edit_label to original (0)
  dec_same_mask_full = tf.einsum("bi,bil->bl", same_mask, rep_enc2dec_full)
  edit_label = tf.where(
      tf.cast(dec_same_mask_full, tf.bool),
      tf.zeros(tf.shape(dec_same_mask_full), dtype=edit_label.dtype),
      edit_label)

  # If same, exclude from LM loss
  dec_same_mask_part = tf.einsum("bi,bij->bj", same_mask, rep_enc2dec_part)
  dec_diff_mask_part = 1.0 - tf.cast(dec_same_mask_part, dec_lm_tgt_mask.dtype)
  dec_lm_tgt_mask = dec_lm_tgt_mask * dec_diff_mask_part

  #### Shared input embedding (for encoder)
  with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    enc_embed = inp_func(inputs=enc_inp, type_id=enc_type)

  #### Encoder TFM
  with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
    enc_tfm_func = _get_tfm_func(initializer, is_training, "pretrain")
    enc_output, _ = enc_tfm_func(
        inputs=enc_embed,
        input_mask=enc_mask,
        perm_mask=None)

  #### Shared input embedding (for decoder)
  with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    dec_embed = inp_func(inputs=dec_inp, type_id=dec_type)

  #### Decoder TFM
  with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    dec_tfm_func = _get_tfm_func(initializer, is_training, "pretrain",
                                 causal=True)
    dec_output, _ = dec_tfm_func(
        inputs=dec_embed,
        input_mask=dec_mask,
        context=enc_output,
        context_mask=enc_mask[:, None, :, None],
        perm_mask=None)

    #### edit type loss
    edit_loss, edit_logits = model.cls_loss(
        hidden=dec_output,
        target=edit_label,
        n_cls=4,
        d_model=FLAGS.d_model,
        initializer=initializer,
        target_mapping=None,
        hidden_mapping=None,
        return_logits=True,
        scope="edit_type_loss")

    dec_tgt_mask = tf.cast(1.0 - dec_mask, tf.float32)
    edit_loss = (tf.reduce_sum(edit_loss * dec_tgt_mask) /
                 tf.reduce_sum(dec_tgt_mask))

    edit_pred = tf.cast(tf.argmax(edit_logits, axis=-1), dtype=edit_label.dtype)
    edit_corr = tf.cast(tf.equal(edit_pred, edit_label), dtype=tf.float32)
    edit_accu = (tf.reduce_sum(edit_corr * dec_tgt_mask) /
                 tf.reduce_sum(dec_tgt_mask))

    if FLAGS.edit_weight > 0:
      # monitor
      monitor_dict["edit_loss"] = edit_loss
      monitor_dict["accu_edit"] = edit_accu

      def get_class_acc(label_id):
        mask = tf.ones(tf.shape(edit_label), dtype=edit_label.dtype) * label_id
        mask = tf.equal(edit_label, mask)
        mask = tf.cast(mask, dtype=dec_tgt_mask.dtype) * dec_tgt_mask
        accu = tf.reduce_sum(edit_corr * mask) / tf.reduce_sum(mask)
        return accu

      monitor_dict["accu_orig"] = get_class_acc(0)
      if FLAGS.del_ratio > 0:
        monitor_dict["accu_del"] = get_class_acc(FLAGS.del_label)
      if FLAGS.ins_ratio > 0:
        monitor_dict["accu_ins"] = get_class_acc(FLAGS.ins_label)
      if FLAGS.rep_ratio > 0:
        monitor_dict["accu_rep"] = get_class_acc(FLAGS.rep_label)

      # accumulate total loss
      total_loss += FLAGS.edit_weight * edit_loss

    #### next-token prediction loss
    if FLAGS.mask_edited_only:
      lm_loss, _ = model.lm_loss(
          hidden=dec_output,
          target=dec_masked_tgt,
          n_token=n_token,
          d_model=FLAGS.d_model,
          initializer=initializer,
          lookup_table=shared_embed_table,
          tie_weight=FLAGS.tie_weight,
          target_mapping=None,
          hidden_mapping=dec_mask_map,
          use_tpu=FLAGS.use_tpu)
      if lm_loss.dtype != tf.float32:
        lm_loss = tf.cast(lm_loss, tf.float32)
      dec_lm_tgt_mask = tf.cast(dec_lm_tgt_mask, lm_loss.dtype)
      lm_loss = (tf.reduce_sum(lm_loss * dec_lm_tgt_mask) /
                 tf.reduce_sum(dec_lm_tgt_mask))
    else:
      lm_loss, _ = model.lm_loss(
          hidden=dec_output,
          target=dec_tgt,
          n_token=n_token,
          d_model=FLAGS.d_model,
          initializer=initializer,
          lookup_table=shared_embed_table,
          tie_weight=FLAGS.tie_weight,
          target_mapping=None,
          hidden_mapping=None,
          use_tpu=FLAGS.use_tpu)
      lm_loss = (tf.reduce_sum(lm_loss * dec_tgt_mask) /
                 tf.reduce_sum(dec_tgt_mask))

    if FLAGS.lm_weight > 0:
      # monitor
      monitor_dict["lm_loss"] = lm_loss

      # accumulate total loss
      total_loss += FLAGS.lm_weight * lm_loss

  return total_loss, {}, monitor_dict


@bf16_decorator
def dae_joint_loss(features, labels, mems, n_token, is_training):
  """DAE loss with generator."""
  del mems
  del labels

  initializer = _get_initializer()
  monitor_dict = {}

  ##### unpack input
  gen_inp = features["gen_inp"]
  gen_tgt = features["gen_tgt"]
  gen_mask_map = features["gen_mask_map"]
  gen_tgt_mask = features["gen_tgt_mask"]

  enc_mask = features["enc_mask"]
  enc_type = features["enc_type"]
  enc_edit_label = features["enc_edit_label"]

  dec_inp = features["dec_inp"]
  dec_tgt = features["dec_tgt"]
  dec_mask = features["dec_mask"]
  dec_type = features["dec_type"]
  edit_label = features["edit_label"]
  dec_mask_map = features["dec_mask_map"]
  dec_masked_tgt = features["dec_masked_tgt"]
  dec_lm_tgt_mask = features["dec_lm_tgt_mask"]
  rep_enc2dec_full = features["rep_enc2dec_full"]
  rep_enc2dec_part = features["rep_enc2dec_part"]

  enc_pos= features["enc_pos"]
  dec_pos = features["dec_pos"]

  if FLAGS.double_type:
    # offer a indicator to differeniate encoder and decoder
    dec_type = dec_type + FLAGS.n_type

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
  same_mask = (tf.cast(same_mask, rep_enc2dec_full.dtype) *
               tf.cast(gen_tgt_mask, rep_enc2dec_full.dtype))
  
  # monitor how many generated tokens are the same as the real ones
  same_prec = (tf.reduce_sum(tf.cast(same_mask, gen_tgt_mask.dtype))
               / tf.reduce_sum(gen_tgt_mask))
  monitor_dict["same_percent"] = same_prec
  
  # If same, change the edit_label to original (0)
  dec_same_mask_full = tf.einsum("bi,bil->bl", same_mask, rep_enc2dec_full)
  edit_label = tf.where(
      tf.cast(dec_same_mask_full, tf.bool),
      tf.zeros(tf.shape(dec_same_mask_full), dtype=edit_label.dtype),
      edit_label)

  # If same, exclude from LM loss
  dec_same_mask_part = tf.einsum("bi,bij->bj", same_mask, rep_enc2dec_part)
  dec_diff_mask_part = 1.0 - tf.cast(dec_same_mask_part, dec_lm_tgt_mask.dtype)
  dec_lm_tgt_mask = dec_lm_tgt_mask * dec_diff_mask_part


  # shapes
  bsz = tf.shape(enc_inp)[0]
  src_len = tf.shape(enc_inp)[1]
  tgt_len = tf.shape(dec_inp)[1]

  ##### format joint model inputs
  inputs = tf.concat([enc_inp, dec_inp], axis=1)
  type_id = tf.concat([enc_type, dec_type], axis=1)
  source_seg = tf.zeros([bsz, src_len], dtype=inputs.dtype)
  target_seg = tf.ones([bsz, tgt_len], dtype=inputs.dtype)

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
      causal_mask
  )
  tgt_mask = tf.concat([tgt_to_src, tgt_to_tgt], axis=2)

  # concat
  perm_mask = tf.concat([src_mask, tgt_mask], axis=1)
  perm_mask = tf.cast(perm_mask, tf_float)

  # padding
  non_pad_mask = tf.not_equal(target_seg, 0)
  all_eos = tf.constant(FLAGS.eos_id, shape=dec_tgt.shape, dtype=dec_tgt.dtype)
  # Replace all <pad> (/P) with <eos> (/S)
  #   - target : /S a1 a2 a3 /S b1 b2 /S c1 c2 /P /P
  #   - tmptgt : /S a1 a2 a3 /S b1 b2 /S c1 c2 /S /S
  tmptgt = tf.where(non_pad_mask, dec_tgt, all_eos)
  # Shift the `tmptgt` to form the (next-step) prediction target
  #   - target   : \S a1 a2 a3 \S b1 b2 \S c1 c2 \P \P
  #   - pred_tgt : a1 a2 a3 \S b1 b2 \S c1 c2 \S \S \S
  pred_tgt = tf.concat([tmptgt[:, 1:], tmptgt[:, :1]], axis=1)
  loss_mask = tf.cast(non_pad_mask, tf.float32)

  pos_seq = tf.concat([enc_pos, dec_pos], axis=1)
  total_loss = 0
  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    input_embed = inp_func(inputs=inputs, pos_seq=pos_seq, type_id=type_id)
    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    output, _ = tfm_func(
        inputs=input_embed,
        input_mask=None,
        perm_mask=perm_mask)

    #### edit loss
    tgt_out = output[:, src_len:]

    edit_loss, edit_logits = model.cls_loss(
        hidden=tgt_out,
        target=edit_label,
        n_cls=4,
        d_model=FLAGS.d_model,
        initializer=initializer,
        target_mapping=None,
        hidden_mapping=None,
        return_logits=True,
        scope="edit_type_loss")

    dec_tgt_mask = tf.cast(1.0 - dec_mask, tf.float32)
    edit_loss = (tf.reduce_sum(edit_loss * dec_tgt_mask) /
                 tf.reduce_sum(dec_tgt_mask))

    edit_pred = tf.cast(tf.argmax(edit_logits, axis=-1), dtype=edit_label.dtype)
    edit_corr = tf.cast(tf.equal(edit_pred, edit_label), dtype=tf.float32)
    edit_accu = (tf.reduce_sum(edit_corr * dec_tgt_mask) /
                 tf.reduce_sum(dec_tgt_mask))

    if FLAGS.edit_weight > 0:
      # monitor
      monitor_dict["edit_loss"] = edit_loss
      monitor_dict["accu_edit"] = edit_accu

      def get_class_acc(label_id):
        mask = tf.ones(tf.shape(edit_label), dtype=edit_label.dtype) * label_id
        mask = tf.equal(edit_label, mask)
        mask = tf.cast(mask, dtype=dec_tgt_mask.dtype) * dec_tgt_mask
        accu = tf.reduce_sum(edit_corr * mask) / tf.reduce_sum(mask)
        return accu

      monitor_dict["accu_orig"] = get_class_acc(0)
      if FLAGS.del_ratio > 0:
        monitor_dict["accu_del"] = get_class_acc(FLAGS.del_label)
      if FLAGS.ins_ratio > 0:
        monitor_dict["accu_ins"] = get_class_acc(FLAGS.ins_label)
      if FLAGS.rep_ratio > 0:
        monitor_dict["accu_rep"] = get_class_acc(FLAGS.rep_label)

      # accumulate total loss
      total_loss += FLAGS.edit_weight * edit_loss

    #### LM loss
    #### Only predict the target part
    tgt_out = output[:, src_len:]
    tf.logging.info("Output: %s, target output: %s",
                    output.shape, tgt_out.shape)
    lm_loss, _ = model.lm_loss(
        hidden=tgt_out,
        target=dec_masked_tgt,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=shared_embed_table,
        tie_weight=FLAGS.tie_weight,
        target_mapping=None,
        hidden_mapping=dec_mask_map,
        use_tpu=FLAGS.use_tpu)      

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)
    dec_lm_tgt_mask = tf.cast(dec_lm_tgt_mask, lm_loss.dtype)
    lm_loss = (tf.reduce_sum(lm_loss * dec_lm_tgt_mask) /
                tf.reduce_sum(dec_lm_tgt_mask))

    if FLAGS.lm_weight > 0:
      # monitor
      monitor_dict["lm_loss"] = lm_loss

      # accumulate total loss
      total_loss += FLAGS.lm_weight * lm_loss


  return total_loss, {}, monitor_dict