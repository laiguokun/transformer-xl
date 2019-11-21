"""Create model function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  import google3.experimental.users.zihangd.pretrain.model as model
  from google3.experimental.users.zihangd.pretrain.model_utils import \
      _get_initializer, _get_inp_func, _get_tfm_func, _get_xlnet_func, \
      bf16_decorator
except ImportError:
  import model
  from model_utils import _get_initializer, _get_inp_func, _get_tfm_func, \
      _get_xlnet_func, bf16_decorator

FLAGS = flags.FLAGS


@bf16_decorator
def get_lm_loss(features, mems, n_token, is_training):
  """LM loss."""
  del mems

  initializer = _get_initializer()

  #### Unpack input
  inputs = features["inputs"]
  target = features["target"]
  type_id = features["type_id"]

  monitor_dict = {}

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    input_embed, word_embed_table = inp_func(
        inputs=inputs, type_id=type_id, return_embed_table=True)

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
def mlm_loss(features, labels, mems, n_token, is_training):
  """Standard MLM loss as in BERT."""
  del labels
  del mems

  initializer = _get_initializer()

  #### Unpack input
  masked_inp = features["masked_input"]
  type_id = features["type_id"]

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
        inputs=masked_inp, type_id=type_id, return_embed_table=True)

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
def electra_loss(features, labels, mems, n_token, is_training):
  """Electra Loss."""
  del mems
  del labels

  initializer = _get_initializer()
  monitor_dict = {}

  #### Unpack input
  original_inp = features["inputs"]
  masked_inp = features["masked_input"]
  type_id = features["type_id"]

  target_mapping = features["target_mapping"]
  target = features["target"]

  gen_inp = masked_inp
  enc_type = type_id
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
        input_mask=None,
        perm_mask=None)

    gen_lm_loss, _, logits = model.lm_loss(
        hidden=gen_output,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=shared_embed_table,
        tie_weight=FLAGS.tie_weight,
        target_mapping=None,
        hidden_mapping=target_mapping,
        return_logits=True,
        use_tpu=FLAGS.use_tpu)

    assert FLAGS.sample_strategy != "token_span"
    if gen_lm_loss.dtype != tf.float32:
      gen_lm_loss = tf.cast(gen_lm_loss, tf.float32)
    gen_loss = tf.reduce_mean(gen_lm_loss)
    monitor_dict["gen_loss"] = gen_loss

  #### Sample from generator
  uniform = tf.random.uniform(minval=0, maxval=1, shape=logits.shape,
                              dtype=tf.float32)
  gumbel = -tf.log(-tf.log(uniform + 1e-9) + 1e-9)
  samples = tf.argmax(logits + gumbel, -1)

  # map `num_predict` samples to full length
  samples = tf.einsum("bm,bml->bl",
                      tf.cast(samples, tf.float32),
                      tf.cast(target_mapping, tf.float32))
  samples = tf.cast(samples, tf.int32)
  enc_inp = tf.where(tf.equal(gen_inp, FLAGS.mask_id), samples, gen_inp)

  is_diff = tf.cast(tf.not_equal(enc_inp, original_inp), tf.float32)
  monitor_dict["diff_percent"] = tf.reduce_mean(is_diff)

  edit_label = tf.cast(tf.equal(enc_inp, original_inp), tf.float32)
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
    edit_loss, edit_corr = model.binary_loss(
        hidden=output,
        target=edit_label,
        d_model=FLAGS.d_model,
        initializer=initializer,
        target_mapping=None,
        hidden_mapping=None)

    edit_loss = tf.reduce_mean(edit_loss)
    edit_accu = tf.reduce_mean(edit_corr * is_diff)

    # monitor
    monitor_dict["edit_loss"] = edit_loss
    monitor_dict["edit_accu_fake"] = edit_accu

    # accumulate total loss
    edit_coeff = 50
    total_loss = edit_coeff * edit_loss + gen_loss

  return total_loss, {}, monitor_dict


@bf16_decorator
def xlnet_loss(features, labels, mems, n_token, is_training):
  del labels
  del mems

  initializer = _get_initializer()

  #### Unpack input
  input_k = features["input_k"]
  input_q = features["input_q"]

  type_id_k = features["type_id_k"]
  type_id_q = features["type_id_q"]

  perm_mask_k = features["perm_mask_k"]
  perm_mask_q = features["perm_mask_q"]

  inputs = tf.concat([input_k, input_q], 1)
  type_id = tf.concat([type_id_k, type_id_q], 1)
  perm_mask = tf.concat([perm_mask_k, perm_mask_q], 1)

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
    input_embed, word_embed_table = inp_func(
        inputs=inputs, type_id=type_id, return_embed_table=True)

    xlnet_func = _get_xlnet_func(initializer, is_training)

    _, output_q, _ = xlnet_func(
        input_k=input_embed[:, :FLAGS.seq_len],
        input_cat=input_embed,
        mapping=target_mapping,
        input_mask=None,
        perm_mask=perm_mask)

    lm_loss, _ = model.lm_loss(
        hidden=output_q,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        target_mapping=None,
        hidden_mapping=None,
        return_logits=False,
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      lm_loss = tf.cast(lm_loss, tf.float32)

    if FLAGS.sample_strategy == "single_token":
      total_loss = tf.reduce_mean(lm_loss)
    else:
      target_mask = tf.cast(target_mask, lm_loss.dtype)
      total_loss = (tf.reduce_sum(lm_loss * target_mask) /
                    tf.reduce_sum(target_mask))
    monitor_dict["lm_loss"] = total_loss

  return total_loss, {}, monitor_dict
