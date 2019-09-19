from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np
import six
from six.moves import zip

import tensorflow as tf


FLAGS = flags.FLAGS


def avg_checkpoints(model_dir, output_model_dir, last_k):
  """Average ckpts."""
  tf.reset_default_graph()

  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  checkpoints = checkpoint_state.all_model_checkpoint_paths[- last_k:]
  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if not name.startswith("global_step"):
      var_values[name] = np.zeros(shape)
  for checkpoint in checkpoints:
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor
    tf.logging.info("Read from checkpoint %s", checkpoint)
  for name in var_values:  # Average.
    var_values[name] /= len(checkpoints)

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  global_step = tf.Variable(
      0, name="global_step", trainable=False, dtype=tf.int64)
  saver = tf.train.Saver(tf.all_variables())

  # Build a model consisting only of variables, set them to the average values.
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(var_values)):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, os.path.join(output_model_dir, "model.ckpt"),
               global_step=global_step)


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


def init_from_checkpoint(global_vars=False):
  tvars = tf.global_variables() if global_vars else tf.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  if FLAGS.init_checkpoint is not None:
    if FLAGS.init_checkpoint.endswith("latest"):
      ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
      init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    else:
      init_checkpoint = FLAGS.init_checkpoint

    tf.logging.info("Initialize from the ckpt %s", init_checkpoint)

    (assignment_map, initialized_variable_names
    ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    if FLAGS.use_tpu:
      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Log customized initialization
    tf.logging.info("**** Global Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
  return scaffold_fn


def configure_tpu(run_internal):
  """Create TPU RunConfig."""

  if FLAGS.use_tpu:
    strategy = None
    tf.logging.info("Use TPU without distribute strategy.")
  elif FLAGS.num_core_per_host == 1:
    strategy = None
    tf.logging.info("Single device mode.")
  else:
    strategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=FLAGS.num_core_per_host)
    tf.logging.info("Use MirroredStrategy with %d devices.",
                    strategy.num_replicas_in_sync)

  # TPU cluster Configuration
  if not run_internal and FLAGS.use_tpu:
    tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster = None

  # Input pipeline
  per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  # Session config
  session_config = tf.ConfigProto(allow_soft_placement=True)
  # Uncomment the following line if you hope to monitor GPU RAM growth
  # session_config.gpu_options.allow_growth = True

  # TPU RunConfig
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster,
      model_dir=FLAGS.model_dir,
      session_config=session_config,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations,
          num_shards=FLAGS.num_hosts * FLAGS.num_core_per_host,
          per_host_input_for_training=per_host_input,
          tpu_job_name=FLAGS.tpu_job_name),
      keep_checkpoint_max=FLAGS.max_save,
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_steps,
      train_distribute=strategy
  )

  return run_config
