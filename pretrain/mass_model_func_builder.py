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
def mass_loss(features, labels, mems, n_token, is_training):
  """MASS pretraining loss."""
  del labels
  del mems

  initializer = _get_initializer()

  # Type
  if FLAGS.use_bfloat16:
    tf_float = tf.bfloat16
  else:
    tf_float = tf.float32

  #### Unpack input
  target = features["target"]
  target_mask = features["target_mask"]
  target_mapping = features["target_mapping"]

  enc_inp = features["enc_inp"]
  enc_type = features["type_id"]
  enc_pos = features["enc_pos"]

  dec_inp = features["dec_inp"]
  dec_type = features["dec_type"]
  dec_pos = features["dec_pos"]
  dec_seg = features["dec_seg"]

  # shapes
  bsz = tf.shape(enc_inp)[0]
  enc_len = tf.shape(enc_inp)[1]
  dec_len = tf.shape(dec_inp)[1]

  ##### format inputs
  inputs = tf.concat([enc_inp, dec_inp], axis=1)
  position = tf.concat([enc_pos, dec_pos], axis=1)
  type_id = tf.concat([enc_type, dec_type], axis=1)

  ##### attention mask: note that `1` indicates CANNOT attend
  # enc mask
  enc_to_enc = tf.zeros(
      [bsz, enc_len, enc_len],
      dtype=tf_float)
  enc_to_dec = tf.ones(
      [bsz, enc_len, dec_len],
      dtype=tf_float)
  enc_mask = tf.concat([enc_to_enc, enc_to_dec], axis=2)

  # dec mask
  dec_to_enc = tf.zeros(
      [bsz, dec_len, enc_len],
      dtype=tf.bool)
  dec_to_dec = tf.not_equal(
      dec_seg[:, :, None],
      dec_seg[:, None, :])
  causal_mask = tf.cast(causal_attn_mask(qlen=dec_len), tf.bool)
  # If any one of them is `1` (indicating cannot attend), i.e. `logical_or`,
  # then the model should NOT attend
  dec_to_dec = tf.logical_or(
      dec_to_dec,
      causal_mask
  )
  dec_mask = tf.cast(tf.concat([dec_to_enc, dec_to_dec], axis=2), tf_float)

  # concat
  perm_mask = tf.concat([enc_mask, dec_mask], axis=1)
  perm_mask = tf.cast(perm_mask, tf_float)

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
    enc_out = output[:, :enc_len]
    dec_out = output[:, enc_len:]
    tf.logging.info("Output: %s, enc output: %s, dec output: %s",
                    output.shape, enc_out.shape, dec_out.shape)

    enc_loss, _ = model.lm_loss(
        hidden=enc_out,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        hidden_mapping=target_mapping,
        use_tpu=FLAGS.use_tpu)

    dec_loss, _ = model.lm_loss(
        hidden=dec_out,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=word_embed_table,
        tie_weight=FLAGS.tie_weight,
        use_tpu=FLAGS.use_tpu)

    if dec_loss.dtype != tf.float32:
      dec_loss = tf.cast(dec_loss, tf.float32)
    if enc_loss.dtype != tf.float32:
      enc_loss = tf.cast(enc_loss, tf.float32)
    loss_mask = tf.cast(target_mask, tf.float32)

    num_loss = tf.reduce_sum(loss_mask)
    avg_dec_loss = tf.reduce_sum(dec_loss * loss_mask) / num_loss
    avg_enc_loss = tf.reduce_sum(enc_loss * loss_mask) / num_loss

  monitor_dict = {}
  total_loss = 0
  if FLAGS.enc_weight > 0:
    total_loss += FLAGS.enc_weight * avg_enc_loss
    monitor_dict["loss_enc"] = avg_enc_loss
  if FLAGS.dec_weight > 0:
    total_loss += FLAGS.dec_weight * avg_dec_loss
    monitor_dict["loss_dec"] = avg_dec_loss

  monitor_dict["loss"] = total_loss

  return total_loss, {}, monitor_dict