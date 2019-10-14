"""Common model utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf


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
