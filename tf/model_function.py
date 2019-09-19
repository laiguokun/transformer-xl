"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  import google3.experimental.users.zihangd.lm.model as model
except ImportError as e:
  import model
# pylint: enable=g-import-not-at-top


flags.DEFINE_integer("log_freq", default=100, help="log frequence.")
flags.DEFINE_integer("max_pos_embed", default=512,
                     help="max pos embedding.")

FLAGS = flags.FLAGS


def construct_scalar_host_call(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None):
  """Construct host call for scalar (rank-0 tensor)."""

  # Only consider scalar (rank-1 tensor)
  metric_names = []
  for k, v in sorted(monitor_dict.items(), key=lambda x: x[0]):
    if v.shape.ndims == 0:
      metric_names.append(k)
      tf.logging.info("Host call receives %s: %s", k, v.shape)
      monitor_dict[k] = tf.reshape(v, [1])
    else:
      tf.logging.info("Host call ignores %s: %s", k, v.shape)

  def host_call_fn(global_step, *args):
    """actual host call function."""
    step = global_step[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          with tf.contrib.summary.record_summaries_every_n_global_steps(
              FLAGS.log_freq, global_step=step):
            tf.contrib.summary.scalar(prefix + name, scalar, step=step)

        return tf.contrib.summary.all_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [monitor_dict[key] for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


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


def _get_tfm_func(n_token, initializer, is_training, **kwargs):
  """choose the correct transformer function."""

  tfm_args = dict(
      n_token=n_token,
      is_training=is_training,
      initializer=initializer,
      attn_type="uni",
      bi_data=False,
      n_layer=FLAGS.n_layer,
      d_model=FLAGS.d_model,
      n_head=FLAGS.n_head,
      d_head=FLAGS.d_head,
      d_inner=FLAGS.d_inner,
      dropout=FLAGS.dropout,
      dropatt=FLAGS.dropatt,
      ff_activation=FLAGS.ff_activation,
      use_tpu=FLAGS.use_tpu,
      use_bfloat16=getattr(FLAGS, "use_bfloat16", False),
  )
  tfm_args.update(kwargs)

  xl_args = dict(
      untie_r=FLAGS.untie_r,
      mem_len=FLAGS.mem_len,
      clamp_len=FLAGS.clamp_len,
      same_length=FLAGS.same_length,
  )
  xl_args.update(tfm_args)
  xl_args.update(kwargs)

  if FLAGS.model_type == "tfm":
    tfm_func = functools.partial(
        model.tfm,
        max_pos_embed=FLAGS.max_pos_embed,
        **tfm_args)
  elif FLAGS.model_type == "tfm_xl":
    tfm_func = functools.partial(
        model.tfm_xl,
        **xl_args)

  return tfm_func


@bf16_decorator
def get_lm_loss(features, mems, is_training):
  """Get LM loss."""

  #### Unpack inputs
  mems = mems.get("mems", None)
  inputs = features["inputs"]
  target = features["target"]

  #### Get transformer function
  n_token = FLAGS.vocab_size
  initializer = _get_initializer()
  tfm_func = _get_tfm_func(n_token=n_token,
                           initializer=initializer,
                           is_training=is_training)

  # new memory
  new_mems = {}

  # tensor to monitor
  monitor_dict = {}

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    # Transformer
    output, new_mems["mems"], lookup_table, _ = tfm_func(
        inputs=inputs,
        mems=mems,
        input_mask=None)

    # LM loss
    lm_loss = model.lm_loss(
        hidden=output,
        target=target,
        n_token=n_token,
        d_model=FLAGS.d_model,
        initializer=initializer,
        lookup_table=lookup_table,
        tie_weight=getattr(FLAGS, "tie_weight", True),
        target_mapping=None,
        hidden_mapping=None,
        use_tpu=FLAGS.use_tpu)

    if lm_loss.dtype != tf.float32:
      tf.logging.info("Cast `lm_loss` to float32 for loss")
      lm_loss = tf.cast(lm_loss, tf.float32)

    total_loss = tf.reduce_mean(lm_loss)
    monitor_dict["lm_loss"] = total_loss
    monitor_dict["ppl"] = tf.exp(total_loss * 1.2)

  return total_loss, new_mems, monitor_dict

