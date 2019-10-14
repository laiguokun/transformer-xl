"""Create model function for estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  import google3.experimental.users.zihangd.pretrain.model as model
except ImportError:
  import model
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


def _get_tfm_func(initializer, is_training, phase, **kwargs):
  """Prepare transformer function."""

  tfm_args = dict(
      n_layer=FLAGS.n_layer,
      d_model=FLAGS.d_model,
      n_head=FLAGS.n_head,
      d_head=FLAGS.d_head,
      d_inner=FLAGS.d_inner,
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

  if FLAGS.attn_to_mask:
    inp_mask = None
  else:
    inp_mask = features["inp_mask"]

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
        input_mask=inp_mask,
        perm_mask=None)

    lm_loss = model.lm_loss(
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
