"""Perform pretraining."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf

  from google3.learning.deepmind.xmanager2.client import xmanager_api  # pylint: disable=unused-import
  import google3.learning.deepmind.xmanager2.client.google as xm  # pylint: disable=unused-import

  import google3.experimental.users.zihangd.pretrain.model_utils as model_utils
  import google3.experimental.users.zihangd.pretrain.optimization as optimization
  import google3.experimental.users.zihangd.pretrain.tpu_estimator_new as tpu_estimator
  import google3.experimental.users.zihangd.pretrain.mlm_model_func_builder as model_func_builder
  import google3.experimental.users.zihangd.pretrain.mlm_input_func_builder as input_func_builder
  from google3.experimental.users.zihangd.pretrain.tokenization import get_tokenizer

  run_internal = True
except ImportError as e:
  import tensorflow as tf
  import model_utils
  import optimization
  import tpu_estimator_new as tpu_estimator
  import mlm_model_func_builder as model_func_builder
  import mlm_input_func_builder as input_func_builder
  from tokenization import get_tokenizer

  run_internal = False
# pylint: enable=g-import-not-at-top


# TPU parameters
flags.DEFINE_string("master", default=None,
                    help="master")
flags.DEFINE_string("tpu", default=None,
                    help="The Cloud TPU to use for training. This should be "
                    "either the name used when creating the Cloud TPU, or a "
                    "grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("gcp_project", default=None,
                    help="Project name for the Cloud TPU-enabled project. If "
                    "not specified, we will attempt to automatically detect "
                    "the GCE project from metadata.")
flags.DEFINE_string("tpu_zone", default=None,
                    help="GCE zone where the Cloud TPU is located in. If not "
                    "specified, we will attempt to automatically detect the "
                    "GCE project from metadata.")
flags.DEFINE_bool("use_tpu", default=True,
                  help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
                     help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=16,
                     help="number of cores per host")
flags.DEFINE_bool("verbose", default=False,
                  help="Whether to print additional information.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("doc_dir", default=None,
                    help="Path to directory containing doc tfrecord.")
flags.DEFINE_string("sent_dir", default=None,
                    help="Path to directory containing sent tfrecord.")
flags.DEFINE_string("semi_dir", default=None,
                    help="Path to directory containing semi-doc tfrecord.")
flags.DEFINE_string("model_dir", default=None,
                    help="Estimator model_dir.")
flags.DEFINE_bool("overwrite", default=False,
                  help="Whether to overwrite exist model dir.")

# Training config
flags.DEFINE_bool("do_train", default=True,
                  help="Whether to run training.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model.")
flags.DEFINE_string("warm_start_path", default=None,
                    help="Checkpoint path for warm start."
                    "If set, will clear Adam states."
                    "Note that the new model_dir should be different "
                    "from `warm_start_path`.")
flags.DEFINE_integer("train_batch_size", default=60,
                     help="Size of train batch.")
flags.DEFINE_integer("train_steps", default=100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
                     help="Number of steps for model checkpointing.")
flags.DEFINE_integer("max_save", default=100000,
                     help="Maximum number of checkpoints to save.")
flags.DEFINE_string("loss_type", default="mlm",
                    help="Type of the loss to use.")

# Evaluation config
flags.DEFINE_bool("do_eval", default=False,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path", default=None,
                    help="Checkpoint path for do_test evaluation."
                    "If set, model_dir will be ignored."
                    "If unset, will use the latest ckpt in `model_dir`.")
flags.DEFINE_integer("eval_batch_size", default=60,
                     help="Size of evalation batch.")
flags.DEFINE_integer("eval_steps", default=None,
                     help="Number of evaluation steps to run.")
flags.DEFINE_integer("start_eval_steps", default=200000,
                     help="Which checkpoint to start with in `do_eval_only` "
                     "mode.")
flags.DEFINE_string("eval_split", default="dev",
                    help="Which data split to evaluate.")

##### Data config
flags.DEFINE_integer("seq_len", default=0,
                     help="tgt len for objective; 0 for not using it")
flags.DEFINE_integer("num_predict", default=None,
                     help="Number of masked tokens.")

##### Loss related
flags.DEFINE_bool("tie_weight", default=True,
                  help="Tie embeddings.")
flags.DEFINE_bool("attn_to_mask", default=True,
                  help="For MLM loss, whether to allow model to attend "
                  "the positions with [mask] tokens.")

##### Precision
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")
flags.DEFINE_bool("float32_softmax", default=True,
                  help="Whether to use float32 softmax.")

##### Monitoring
flags.DEFINE_integer("log_freq", default=100, help="log frequence.")

FLAGS = flags.FLAGS


def metric_fn(loss):
  """Evaluation metric Fn which runs on CPU."""
  perplexity = tf.exp(tf.reduce_mean(loss))
  return {
      "eval/loss": tf.metrics.mean(loss),
      "eval/perplexity": tf.metrics.mean(perplexity),
  }


def get_model_fn(n_token):
  """doc."""
  def model_fn(features, labels, mode, params):
    """doc."""
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Retrieve `mems` from `params["cache"]`
    mems = {}
    idx = 0
    for obj_len, key in zip([FLAGS.seq_len], ["mems"]):
      if obj_len > 0 and FLAGS.mem_len > 0:
        n_layer = FLAGS.n_layer
        if FLAGS.use_extra_layer:
          n_layer += 1
        mems[key] = params["cache"][idx*n_layer: (idx+1)*n_layer]
        idx += 1

    #### Get loss from inputs
    if FLAGS.loss_type == "electra":
      total_loss, new_mems, monitor_dict = model_func_builder.electra_loss(
          features, labels, mems, n_token, is_training)
    elif FLAGS.loss_type == "mlm":
      total_loss, new_mems, monitor_dict = model_func_builder.mlm_loss(
          features, labels, mems, n_token, is_training)
    else:
      raise NotImplementedError

    #### Turn `new_mems` into `new_cache`
    new_cache = []
    for obj_len, key in zip([FLAGS.seq_len], ["mems"]):
      if obj_len > 0 and FLAGS.mem_len > 0:
        new_cache += new_mems[key]

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: %d", num_params)

    if FLAGS.verbose:
      format_str = "{{:<{0}s}}\t{{}}".format(
          max([len(v.name) for v in tf.trainable_variables()]))
      for v in tf.trainable_variables():
        tf.logging.info(format_str.format(v.name, v.get_shape()))

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      #### Reduce sum losses from all TPU cores
      with tf.colocate_with(total_loss):
        total_loss = tf.contrib.tpu.cross_replica_sum(total_loss)
        total_loss = total_loss / FLAGS.num_hosts / FLAGS.num_core_per_host
      metric_loss = tf.reshape(total_loss, [1])

      #### Constructing evaluation TPUEstimatorSpec with new cache.
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [metric_loss]))

      eval_spec.cache = new_cache

      return eval_spec

    #### Get the train op
    train_op, optim_dict = optimization.get_train_op(total_loss)
    monitor_dict.update(optim_dict)

    #### Customized initial checkpoint
    tvars = tf.global_variables()
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
      ) = model_utils.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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

    #### Creating host calls
    host_call = model_utils.construct_scalar_host_call(
        monitor_dict=monitor_dict,
        model_dir=FLAGS.model_dir,
        prefix="train/",
        reduce_fn=tf.reduce_mean,
        log_freq=FLAGS.log_freq)

    #### Constructing training TPUEstimatorSpec with new cache.
    train_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
        scaffold_fn=scaffold_fn)

    train_spec.cache = new_cache

    return train_spec

  return model_fn


def get_cache_fn(mem_len):
  """doc."""
  tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32
  def cache_fn(batch_size):
    """Call back function used to create initial cache in TPUEstimator."""
    mems = []
    for obj_len in [FLAGS.seq_len]:
      if obj_len > 0:
        for _ in range(FLAGS.n_layer + int(FLAGS.use_extra_layer)):
          zeros = tf.zeros(
              [batch_size, mem_len, FLAGS.d_model],
              dtype=tf_float)
          mems.append(zeros)

    return mems

  if mem_len > 0:
    return cache_fn
  else:
    return None


def get_input_fn(split):
  """doc."""
  if split == "train":
    batch_size = FLAGS.train_batch_size
  else:
    batch_size = FLAGS.eval_batch_size

  kwargs = dict(
      doc_dir=FLAGS.doc_dir,
      semi_dir=FLAGS.semi_dir,
      sent_dir=FLAGS.sent_dir,
      split=split,
      uncased=FLAGS.uncased,
      seq_len=FLAGS.seq_len,
      num_predict=FLAGS.num_predict,
      bsz_per_host=batch_size // FLAGS.num_hosts,
      num_hosts=FLAGS.num_hosts,
      num_core_per_host=FLAGS.num_core_per_host,
      use_bfloat16=FLAGS.use_bfloat16,
  )

  input_fn = input_func_builder.get_input_fn(**kwargs)

  return input_fn


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate FLAGS
  if FLAGS.save_steps == 0:
    FLAGS.save_steps = None

  assert FLAGS.seq_len > 0

  #### Tokenizer
  tokenizer = get_tokenizer()

  #### Get corpus info
  n_token = tokenizer.get_vocab_size()
  tf.logging.info("n_token %d", n_token)

  if FLAGS.do_train:
    # Get train input function
    train_input_fn = get_input_fn("train")

    # Get train cache function
    train_cache_fn = get_cache_fn(FLAGS.mem_len)
  else:
    train_cache_fn = None

  if FLAGS.do_eval:
    assert FLAGS.num_hosts == 1
    # Get eval input function
    eval_input_fn = get_input_fn(FLAGS.eval_split)
    tf.logging.info("num of eval batches %d", FLAGS.eval_steps)

    # Get eval cache function
    eval_cache_fn = get_cache_fn(FLAGS.mem_len)
  else:
    eval_cache_fn = None

  ##### Get model function
  model_fn = get_model_fn(n_token)

  ##### Create TPUEstimator
  # TPU Configuration
  if not run_internal and FLAGS.use_tpu:
    tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster = None

  per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(allow_soft_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations,
          per_host_input_for_training=per_host_input),
      keep_checkpoint_max=FLAGS.max_save,
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_steps
  )

  # warm start
  warm_start_from = None
  if FLAGS.warm_start_path is not None:
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAGS.warm_start_path)

  # TPU Estimator
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      train_cache_fn=train_cache_fn,
      eval_cache_fn=eval_cache_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params={},
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_on_tpu=FLAGS.use_tpu,
      warm_start_from=warm_start_from)

  #### Training
  if FLAGS.do_train:
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  #### Evaluation
  if FLAGS.do_eval:
    if FLAGS.eval_ckpt_path is not None:
      if FLAGS.eval_ckpt_path.endswith("latest"):
        ckpt_dir = os.path.dirname(FLAGS.eval_ckpt_path)
        FLAGS.eval_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

      ret = estimator.evaluate(input_fn=eval_input_fn,
                               steps=FLAGS.eval_steps,
                               checkpoint_path=FLAGS.eval_ckpt_path)
      tf.logging.info("=" * 200)
      log_str = "Eval results | "
      for key, val in ret.items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)
    else:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
      eval_results = []
      for eval_checkpoint in ckpt_state.all_model_checkpoint_paths:
        if not tf.gfile.Exists(eval_checkpoint + ".index"):
          continue
        global_step = int(eval_checkpoint.split("-")[-1])
        if (global_step < FLAGS.start_eval_steps or global_step >
            FLAGS.train_steps):
          continue
        tf.logging.info("Evaluate ckpt %d", global_step)
        ret = estimator.evaluate(input_fn=eval_input_fn,
                                 steps=FLAGS.eval_steps,
                                 checkpoint_path=eval_checkpoint)
        eval_results.append(ret)

      eval_results.sort(key=lambda x: x["perplexity"])

      tf.logging.info("=" * 200)
      log_str = "Best results | "
      for key, val in eval_results[0].items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)


if __name__ == "__main__":
  app.run(main)
