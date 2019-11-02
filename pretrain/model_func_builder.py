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
except ImportError:
  import model
  from common_ops import causal_attn_mask
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS


def bf16_decorator(func):
  @functools.wraps(func)
  def wrapped_func(*args, **kwargs):
    if FLAGS.use_bfloat16:
      with tf.tpu.bfloat16_scope():
        return func(*args, **kwargs)
    else:
      return func(*args, **kwargs)

  return wrapped_func


def _get_initializer():
  """Get variable intializer."""
  tf.logging.info("Using %s initializer", FLAGS.init)
  if FLAGS.init == "uniform":
    initializer = tf.initializers.random_uniform(
        minval=-FLAGS.init_range,
        maxval=FLAGS.init_range,
        seed=None)
  elif FLAGS.init == "normal":
    initializer = tf.initializers.random_normal(
        stddev=FLAGS.init_std,
        seed=None)
  elif FLAGS.init == "truncated_normal":
    initializer = tf.initializers.truncated_normal(
        stddev=FLAGS.init_std,
        seed=None)
  else:
    raise ValueError("Initializer {} not supported".format(FLAGS.init))
  return initializer


def _get_inp_func(n_token, d_embed, initializer, is_training, **kwargs):
  """Prepare input function."""
  if FLAGS.input_proc == "inv_sqrt":
    inp_func = functools.partial(
        model.mt_input,
        n_token=n_token,
        n_type=FLAGS.n_type,
        d_embed=d_embed,
        dropout=FLAGS.dropout,
        clamp_len=FLAGS.clamp_len,
        initializer=initializer,
        is_training=is_training,
        use_tpu=FLAGS.use_tpu,
        **kwargs)
  elif FLAGS.input_proc == "layer_norm":
    inp_func = functools.partial(
        model.input_layer,
        n_token=n_token,
        n_type=FLAGS.n_type,
        n_pos=FLAGS.max_pos_len,
        d_embed=d_embed,
        dropout=FLAGS.dropout,
        initializer=initializer,
        is_training=is_training,
        use_tpu=FLAGS.use_tpu,
        **kwargs)
  else:
    raise NotImplementedError

  return inp_func


def _get_tfm_func(initializer, is_training, phase, shrink=1, **kwargs):
  """Prepare transformer function."""

  tfm_args = dict(
      n_layer=FLAGS.n_layer,
      d_model=FLAGS.d_model // shrink,
      n_head=FLAGS.n_head // shrink,
      d_head=FLAGS.d_head,
      d_inner=FLAGS.d_inner // shrink,
      dropout=FLAGS.dropout,
      dropatt=FLAGS.dropatt,
      dropact=FLAGS.dropact,
      ff_activation=FLAGS.ff_activation,
      initializer=initializer,
      is_training=is_training,
  )
  tfm_args.update(kwargs)

  if phase == "pretrain":
    xl_args = dict(
        untie_r=FLAGS.untie_r,
        mem_len=FLAGS.mem_len,
    )
  elif phase == "finetune":
    xl_args = dict(
        untie_r=FLAGS.untie_r,
        mem_len=None,
    )
  else:
    raise ValueError("Unsupported phase {}".format(phase))
  xl_args.update(tfm_args)
  xl_args.update(kwargs)

  if FLAGS.model_type == "tfm":
    tfm_func = functools.partial(
        model.transformer,
        **tfm_args)
  elif FLAGS.model_type == "tfm_xl":
    tfm_func = functools.partial(
        model.transformer_xl,
        **xl_args)

  return tfm_func


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


def get_loss(features, labels, mems, n_token, is_training):
  """Loss selector."""
  if FLAGS.loss_type == "mlm":
    return mlm_loss(features, labels, mems, n_token, is_training)
  else:
    raise NotImplementedError


def extract_hiddens(inputs, type_id, n_token, is_training):
  """Extract all hidden states."""
  initializer = _get_initializer()

  #### Transformer Model
  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    embeddings = inp_func(inputs=inputs, type_id=type_id)

    tfm_func = _get_tfm_func(initializer,
                             is_training,
                             phase="pretrain")
    hiddens, _ = tfm_func(
        inputs=embeddings,
        input_mask=None,
        perm_mask=None,
        return_all_hidden=True)

    return hiddens


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
  dec_mask_tgt = features["dec_mask_tgt"]
  dec_lm_tgt_mask = features["dec_lm_tgt_mask"]

  #### Shared input embedding (for generator)
  with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    inp_func = _get_inp_func(n_token,
                             FLAGS.d_model,
                             initializer,
                             is_training)
    gen_embed, shared_embed_table = inp_func(
        inputs=gen_inp, type_id=None, return_embed_table=True)

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

  # map `num_predict` samples to full length
  samples = tf.einsum("bm,bml->bl",
                      tf.cast(samples, tf.float32),
                      tf.cast(gen_mask_map, tf.float32))
  samples = tf.cast(samples, tf.int32)
  enc_inp = tf.where(tf.equal(gen_inp, FLAGS.mask_id), samples, gen_inp)

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
        context_mask=enc_mask,
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

    # monitor
    monitor_dict["edit_loss"] = edit_loss
    monitor_dict["edit_accu"] = edit_accu

    # accumulate total loss
    total_loss += FLAGS.edit_weight * edit_loss

    #### next-token prediction loss
    if FLAGS.mask_edited_only:
      lm_loss, _ = model.lm_loss(
          hidden=dec_output,
          target=dec_mask_tgt,
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

    # monitor
    monitor_dict["lm_loss"] = lm_loss

    # accumulate total loss
    total_loss += lm_loss

  return total_loss, {}, monitor_dict


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
  src_type_id = tf.zeros([bsz, src_len], dtype=inputs.dtype)
  tgt_type_id = tf.ones([bsz, tgt_len], dtype=inputs.dtype)
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
      causal_mask
  )
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
