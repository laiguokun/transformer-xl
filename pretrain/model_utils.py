"""Common model utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import re

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


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      tf.logging.info("Ignore variable %s", name)
      continue
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def construct_scalar_host_call(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None,
    log_freq=100):
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
              log_freq, global_step=step):
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


def _get_inp_func(n_token, d_embed, initializer, is_training, **kwargs):
  """Prepare input function."""
  if FLAGS.double_type:
    n_type = FLAGS.n_type * 2
  else:
    n_type = FLAGS.n_type
  if FLAGS.input_proc == "inv_sqrt":
    inp_func = functools.partial(
        model.mt_input,
        n_token=n_token,
        n_type=n_type,
        d_embed=d_embed,
        dropout=FLAGS.dropout,
        clamp_len=FLAGS.clamp_len,
        initializer=initializer,
        is_training=is_training,
        use_tpu=FLAGS.use_tpu,
        rel_attn=FLAGS.rel_attn,
        **kwargs)
  elif FLAGS.input_proc == "layer_norm":
    inp_func = functools.partial(
        model.input_layer,
        n_token=n_token,
        n_type=n_type,
        n_pos=FLAGS.max_pos_len,
        d_embed=d_embed,
        dropout=FLAGS.dropout,
        initializer=initializer,
        is_training=is_training,
        use_tpu=FLAGS.use_tpu,
        rel_attn=FLAGS.rel_attn,
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
      rel_attn=FLAGS.rel_attn,
      clamp_len=FLAGS.clamp_len,
      initializer=initializer,
      is_training=is_training,
  )
  tfm_args.update(kwargs)

  if phase == "pretrain":
    xl_args = dict(
        mem_len=FLAGS.mem_len,
    )
  elif phase == "finetune":
    xl_args = dict(
        mem_len=None,
    )
  else:
    raise ValueError("Unsupported phase {}".format(phase))
  xl_args.update(tfm_args)
  xl_args.update(kwargs)

  tfm_func = functools.partial(
      model.transformer,
      **tfm_args)

  return tfm_func


def _get_xlnet_func(initializer, is_training, shrink=1, **kwargs):
  """Prepare transformer function for two-stream xlnet loss."""
  xlnet_args = dict(
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
      clamp_len=FLAGS.clamp_len,
  )
  xlnet_args.update(kwargs)
  xlnet_func = functools.partial(model.xlnet, **xlnet_args)

  return xlnet_func


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
