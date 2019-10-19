"""Optimization related functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

from six.moves import zip

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from google3.experimental.users.zihangd.pretrain import tpu_optimizer
except ImportError as e:
  import tpu_optimizer
# pylint: enable=g-import-not-at-top


##### Optimization related flags #####
flags.DEFINE_string("optimizer", default="adamw", help="Optimizer to use")

# learning rate schedule
flags.DEFINE_float("init_lr", default=0, help="initial learning rate.")
flags.DEFINE_float("learning_rate", default=1e-5, help="maximum learning rate")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")

# weight decay
flags.DEFINE_float("weight_decay", default=0.00, help="weight decay rate")

# gradient compute and clip
flags.DEFINE_bool("bert_grad", False,
                  help="Whether to use tf.gradients as used in BERT.")
flags.DEFINE_bool("clip_each_core", default=True,
                  help="Clip gradient on each core.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_bool("per_core_clip", True,
                  help="Perform gradient clip on each TPU core.")
flags.DEFINE_bool("skip_nan_grad", False,
                  help="Whether to use skip NaN or Inf gradient.")

# used during finetune
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")

# adam specific hparams
flags.DEFINE_float("adam_beta1", default=0.9,
                   help="The exponential decay rate for the 1st moment.")
flags.DEFINE_float("adam_beta2", default=0.999,
                   help="The exponential decay rate for the 2nd moment.")
flags.DEFINE_bool("adam_correction", default=True,
                  help="Use the adam bias correction.")
flags.DEFINE_bool("use_wd_exclusion", default=False,
                  help="Exclude certain params from weight decay as in BERT.")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="adam epsilon")

# optimizer with momentum
flags.DEFINE_float("momentum", default=0.99,
                   help="Gradient clipping value.")

# debug optimization
flags.DEFINE_bool("log_all_gnorm", False,
                  help="Whether to use monitor gradient norms of all vars.")

FLAGS = flags.FLAGS


def _get_variable_name(param_name):
  """Get the variable name from the tensor name."""
  m = re.match("^(.*):\\d+$", param_name)
  if m is not None:
    param_name = m.group(1)
    return param_name


def get_train_op(total_loss):
  """Get the train op from training loss."""
  monitor_dict = {}

  ##### Configure optimizer
  global_step = tf.train.get_or_create_global_step()

  # Warmup the learning rate linearly
  if FLAGS.warmup_steps > 0:
    warmup_lr = (tf.cast(global_step, tf.float32)
                 / tf.cast(FLAGS.warmup_steps, tf.float32)
                 * (FLAGS.learning_rate - FLAGS.init_lr)) + FLAGS.init_lr
  else:
    warmup_lr = 0.0

  # Decay the learning rate
  if FLAGS.decay_method == "poly":
    decay_lr = tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        end_learning_rate=FLAGS.learning_rate * FLAGS.min_lr_ratio)
  elif FLAGS.decay_method == "cos":
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)
  elif FLAGS.decay_method == "inv_sqrt":
    global_step_f32 = tf.cast(global_step, tf.float32)
    decay_lr = (FLAGS.learning_rate * (FLAGS.warmup_steps ** 0.5) *
                (global_step_f32 ** -0.5))
  else:
    raise ValueError(FLAGS.decay_method)

  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  if (FLAGS.weight_decay > 0 and not FLAGS.use_tpu and
      FLAGS.num_core_per_host > 1):
    raise ValueError("Do not support `weight_decay > 0` with multi-gpu "
                     "training so far.")

  if FLAGS.optimizer == "adamw":
    if FLAGS.use_wd_exclusion:
      exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"]
    else:
      exclude_from_weight_decay = []

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        beta_1=FLAGS.adam_beta1,
        beta_2=FLAGS.adam_beta2,
        epsilon=FLAGS.adam_epsilon,
        bias_correction=FLAGS.adam_correction,
        exclude_from_weight_decay=exclude_from_weight_decay,
        weight_decay_rate=FLAGS.weight_decay)
  elif FLAGS.optimizer == "adam":
    assert FLAGS.weight_decay == 0, "Use adamw for weight decay"
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.adam_epsilon)
  elif FLAGS.optimizer == "lamb":
    if FLAGS.adam_correction:
      tf.logging.info("Ignore adam correction in LAMB.")
    if FLAGS.use_wd_exclusion:
      exclude_from_weight_decay = ["LayerNorm", "layer_norm", "bias"]
    else:
      exclude_from_weight_decay = []

    optimizer = LAMBOptimizer(
        learning_rate=learning_rate,
        beta_1=FLAGS.adam_beta1,
        beta_2=FLAGS.adam_beta2,
        epsilon=FLAGS.adam_epsilon,
        exclude_from_weight_decay=exclude_from_weight_decay,
        weight_decay_rate=FLAGS.weight_decay)
  elif FLAGS.optimizer == "momentum":
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=FLAGS.momentum,
    )
  elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        momentum=FLAGS.momentum,
    )

  if FLAGS.use_tpu:
    if FLAGS.per_core_clip:
      optimizer = tpu_optimizer.CrossShardOptimizer(
          optimizer, skip_nan_grad=FLAGS.skip_nan_grad)
    else:
      optimizer = tpu_optimizer.CrossShardOptimizer(
          optimizer, skip_nan_grad=FLAGS.skip_nan_grad, clip=FLAGS.clip)

  ##### Compute gradient
  if FLAGS.bert_grad:
    tf.logging.info("Use bert grad")
    variables = tf.trainable_variables()
    gradients = tf.gradients(total_loss, variables)
  else:
    tf.logging.info("Do not use bert grad")
    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)

  if FLAGS.clip > 0 and FLAGS.per_core_clip:
    tf.logging.info("Clip local gradient with norm %.3f.", FLAGS.clip)
    clipped, local_gnorm = tf.clip_by_global_norm(gradients, FLAGS.clip)
  else:
    tf.logging.info("Do not clip local gradient.")
    clipped = list(gradients)
    local_gnorm = tf.global_norm(gradients)

  if FLAGS.log_all_gnorm:
    for clip, grad, var in zip(clipped, gradients, variables):
      if grad is not None:
        var_name = _get_variable_name(var.name)
        # monitor_dict[var_name] = tf.norm(var)
        monitor_dict[var_name + "_grad"] = tf.norm(grad)
        monitor_dict[var_name + "_clip"] = tf.norm(clip)

  # layer-wise learning rate decay
  if getattr(FLAGS, "lr_layer_decay_rate", 1.0) != 1.0:
    def _get_layer_id(name):
      m = re.search(r"model/transformer/layer_(\d+?)/", name)
      if not m: return None
      return int(m.group(1))

    n_layer = 0
    for i in range(len(clipped)):
      layer_id = _get_layer_id(variables[i].name)
      if layer_id is None: continue
      n_layer = max(n_layer, layer_id + 1)

    abs_rate_vector = FLAGS.lr_layer_decay_rate ** tf.reverse(
        tf.range(n_layer, dtype=clipped[0].dtype), axis=[0])

    for i in range(len(clipped)):
      layer_id = _get_layer_id(variables[i].name)
      if layer_id is not None:
        abs_rate = FLAGS.lr_layer_decay_rate ** (n_layer - 1 - layer_id)
        clipped[i] *= abs_rate
        tf.logging.info("Apply mult %.4f to layer-%d grad of %s",
                        abs_rate, layer_id, variables[i].name)
      elif "_bias" in variables[i].name:
        if len(clipped[i].get_shape()) == 3:
          clipped[i] *= abs_rate_vector[:, None, None]
      elif "seg_embed" in variables[i].name:
        if len(clipped[i].get_shape()) == 4:
          clipped[i] *= abs_rate_vector[:, None, None, None]
      elif "lookup_table" in variables[i].name:
        abs_rate = FLAGS.lr_layer_decay_rate ** n_layer
        if isinstance(clipped[i], tf.IndexedSlices):
          clipped[i] = tf.IndexedSlices(clipped[i].values * abs_rate,
                                        clipped[i].indices,
                                        clipped[i].dense_shape)
        else:
          clipped[i] *= abs_rate

      else:
        tf.logging.info("Grad of %s is not decayed.", variables[i].name)

  ##### Construct train op
  if FLAGS.use_tpu:
    train_op, global_gnorm = optimizer.apply_gradients(
        list(zip(clipped, variables)), global_step=global_step)
    monitor_dict["global_gnorm"] = global_gnorm
  else:
    train_op = optimizer.apply_gradients(
        list(zip(clipped, variables)), global_step=global_step)

  # Manually increment `global_step` for AdamW and LAMB
  if FLAGS.optimizer in ["adamw", "lamb"]:
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  monitor_dict["local_gnorm"] = local_gnorm
  monitor_dict["learning_rate"] = learning_rate

  return train_op, monitor_dict


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               bias_correction=False,
               exclude_from_weight_decay=None,
               include_in_weight_decay=[
                   "r_s_bias", "r_r_bias", "r_w_bias",
                   "gamma_private", "beta_private"],
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.bias_correction = bias_correction
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.include_in_weight_decay = include_in_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []

    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn"t interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name) and self.weight_decay_rate > 0:
        update += self.weight_decay_rate * param

      # Adam bias correction
      if self.bias_correction:
        global_step_float = tf.cast(global_step, update.dtype)
        bias_correction1 = 1.0 - self.beta_1 ** (global_step_float + 1)
        bias_correction2 = 1.0 - self.beta_2 ** (global_step_float + 1)
        learning_rate = (self.learning_rate * tf.sqrt(bias_correction2)
                         / bias_correction1)
      else:
        learning_rate = self.learning_rate

      update_with_lr = learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])

    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False

    for r in self.include_in_weight_decay:
      if re.search(r, param_name) is not None:
        tf.logging.info("Include %s in weight decay", param_name)
        return True

    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          tf.logging.info("Adam WD excludes %s", param_name)
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               include_in_weight_decay=[
                   "r_s_bias", "r_r_bias", "r_w_bias",
                   "gamma_private", "beta_private"],
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.include_in_weight_decay = include_in_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    # TODO(jingli): validate if exclude_from_layer_adaptation is necessary.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn"t interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.norm(param, ord=2)
        g_norm = tf.norm(update, ord=2)
        ratio = tf.where(tf.greater(w_norm, 0), tf.where(
            tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      update_with_lr = ratio * self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    for r in self.include_in_weight_decay:
      if re.search(r, param_name) is not None:
        tf.logging.info("Include %s in weight decay", param_name)
        return True
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          tf.logging.info("Exclude %s from weight decay", param_name)
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.include_in_weight_decay:
        if re.search(r, param_name) is not None:
          tf.logging.info("Include %s in layer adaption", param_name)
          return True

      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          tf.logging.info("Exclude %s from layer adaption", param_name)
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

