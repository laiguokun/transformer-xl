"""Pretraining on TPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np

# pylint: disable=g-import-not-at-top
try:
  import tensorflow.google as tf
  import google3.experimental.users.zihangd.lm.data_utils as data_utils
  import google3.experimental.users.zihangd.lm.input_function as input_function
  import google3.experimental.users.zihangd.lm.model_function as model_function
  import google3.experimental.users.zihangd.lm.model_utils as model_utils
  import google3.experimental.users.zihangd.lm.optimization as optimization
  import google3.experimental.users.zihangd.lm.tokenization as tokenization
  import google3.experimental.users.zihangd.lm.tpu_estimator_new as tpu_estimator
  run_internal = True
except ImportError as e:
  import tensorflow as tf
  import data_utils
  import input_function
  import model_function
  import model_utils
  import optimization
  import tokenization
  import tpu_estimator_new as tpu_estimator
  run_internal = False
# pylint: enable=g-import-not-at-top


# TPU parameters
flags.DEFINE_string("master", default=None,
                    help="master")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
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
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=10,
                     help="Number of passed used for training.")
flags.DEFINE_string("record_dir", default=None,
                    help="Path to directory containing tfrecords.")
flags.DEFINE_string("model_dir", default=None,
                    help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="Checkpoint path for initializing the model.")

# Training config
flags.DEFINE_bool("do_train", True,
                  help="Whether to perform training.")
flags.DEFINE_integer("train_batch_size", default=16,
                     help="Size of the train batch across all hosts.")
flags.DEFINE_integer("train_steps", default=100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Number of steps for model checkpointing. "
                     "None for not saving checkpoints")
flags.DEFINE_integer("max_save", default=100000,
                     help="Maximum number of checkpoints to save.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")
flags.DEFINE_bool("float32_softmax", True,
                  help="Whether to use float32 for softmax.")

# Evaluation config
flags.DEFINE_bool("do_eval", False,
                  help="Whether to perform evaluation.")
flags.DEFINE_enum("eval_split", "valid", ["valid", "test"],
                  help="Which split to eval.")
flags.DEFINE_integer("eval_batch_size", default=16,
                     help="Size of the eval batch across all hosts.")
flags.DEFINE_string("eval_ckpt_path", default=None,
                    help="Path to the specific ckpt to evaluate.")
flags.DEFINE_integer("eval_steps", default=None,
                     help="Number of evaluation steps to run.")

# Data config
flags.DEFINE_string("tokenizer_type", "sent_piece",
                    help="Type of the tokenizer.")
flags.DEFINE_string("tokenizer_paths", "",
                    help="Comma separated string.")
flags.DEFINE_bool("lower_case", False,
                  help="Use lower_case inputs or not.")
flags.DEFINE_integer("seq_len", default=512,
                     help="Sequence length for pretraining.")

# Model config
flags.DEFINE_integer("mem_len", default=0,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
                     help="Dimension of inner hidden size in positionwise "
                     "feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=True,
                  help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("model_type", default="tfm_xl",
                    help="Model type.")
flags.DEFINE_string("ff_activation", default="gelu",
                    help="Activation type used in position-wise feed-forward.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform", "truncated_normal"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def get_model_fn():
  """doc."""
  def model_fn(features, labels, mode, params):
    """doc."""
    # not used
    del labels

    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Retrieve `mems` from `params["cache"]`
    mems = None
    if FLAGS.mem_len > 0:
      mems = params["cache"]

    #### Get loss from inputs
    total_loss, new_mems, monitor_dict = model_function.get_lm_loss(
        features, mems, is_training)

    #### Put `new_mems` into `new_cache`
    new_cache = []
    if FLAGS.mem_len > 0:
      new_cache += new_mems

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: %d", num_params)

    if mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(loss):
        """Evaluation metric Fn which runs on CPU."""
        perplexity = tf.exp(tf.reduce_mean(loss) * 1.2)
        return {
            "perplexity": tf.metrics.mean(perplexity),
        }

      metric_loss = tf.reshape(total_loss, [1, 1])
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [metric_loss]))

      eval_spec.cache = new_cache

      return eval_spec

    #### Configuring the optimizer
    train_op, optim_dict = optimization.get_train_op(total_loss)
    monitor_dict.update(optim_dict)

    #### Customized initial checkpoint
    scaffold_fn = model_utils.init_from_checkpoint(global_vars=True)

    #### Creating host calls
    host_call = model_function.construct_scalar_host_call(
        monitor_dict=monitor_dict,
        model_dir=FLAGS.model_dir,
        prefix="train/",
        reduce_fn=tf.reduce_mean)

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
    mems = []
    if FLAGS.mem_len > 0:
      for _ in range(FLAGS.n_layer):
        if FLAGS.chunk_len is not None:
          zeros = tf.zeros(
              [batch_size, mem_len, FLAGS.chunk_len, FLAGS.d_model],
              dtype=tf_float)
        else:
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

  input_fn, record_info_dict = input_function.get_input_fn(
      tfrecord_dir=FLAGS.record_dir,
      split=split,
      bsz_per_host=batch_size // FLAGS.num_hosts,
      seq_len=FLAGS.seq_len,
      num_hosts=FLAGS.num_hosts,
      num_core_per_host=FLAGS.num_core_per_host,
      lower_case=FLAGS.lower_case,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16)

  return input_fn, record_info_dict


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.get_tokenizer(
      tokenizer_type=FLAGS.tokenizer_type,
      paths=FLAGS.tokenizer_paths.split(","),
      do_lower_case=FLAGS.lower_case)
  data_utils.setup_special_ids(tokenizer)

  ##### Get train cache function
  train_cache_fn = get_cache_fn(FLAGS.mem_len)
  eval_cache_fn = get_cache_fn(FLAGS.mem_len)

  ##### Get model function
  model_fn = get_model_fn()

  ##### Create TPUEstimator
  # TPU Configuration
  run_config = model_utils.configure_tpu(run_internal)

  # TPU Estimator
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      train_cache_fn=train_cache_fn,
      eval_cache_fn=eval_cache_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_on_tpu=FLAGS.use_tpu)

  ##### Training
  if FLAGS.do_train:
    # Get train input function
    train_input_fn, train_record_info_dict = get_input_fn("train")
    tf.logging.info("num of train batches %d",
                    train_record_info_dict["num_batch"])

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  #### Evaluation
  if FLAGS.do_eval:
    # Get eval input function
    eval_input_fn, eval_record_info_dict = get_input_fn(FLAGS.eval_split)
    num_eval_batch = eval_record_info_dict["num_batch"]
    tf.logging.info("num of eval batches %d", num_eval_batch)

    if FLAGS.eval_steps is not None:
      eval_steps = min(num_eval_batch, FLAGS.eval_steps)
    else:
      eval_steps = num_eval_batch

    estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps,
                       checkpoint_path=FLAGS.eval_ckpt_path)


if __name__ == "__main__":
  app.run(main)
