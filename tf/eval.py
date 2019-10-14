from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

import data_utils
import tokenization
import input_function
import model_function

from gpu_utils import assign_to_gpu, average_grads_and_vars
from renormalize_calc_np import RenormalizeVocabPost

import numpy as np

# GPU config
flags.DEFINE_bool("use_tpu", default=False,
                  help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
                     help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=1,
                     help="Number of cores per host")

# Renormalization
flags.DEFINE_bool("renormalize", default=False,
                  help="Use TPUs rather than plain CPUs.")
flags.DEFINE_string("vocab_dir", default="",
                    help="Path to vocab file directory.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_dir", default="",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=True,
      help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False,
      help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path", None,
      help="Checkpoint path for do_test evaluation."
           "If set, model_dir will be ignored."
           "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
      help="Checkpoint path for warm start."
           "If set, will clear Adam states."
           "Note that the new model_dir should be different"
           " from warm_start_path.")

# Optimization config
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=60,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=60,
      help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=500,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
      help="number of steps for model checkpointing.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False,
      help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False,
      help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
      help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split", "valid",
      help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("seq_len", default=512,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=512,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_string("tokenizer_type", "sent_piece",
                    help="Type of the tokenizer.")
flags.DEFINE_string("tokenizer_paths", "",
                    help="Comma separated string.")
flags.DEFINE_bool("lower_case", False,
                  help="Use lower_case inputs or not.")


flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=True,
      help="untie r_w_bias and r_r_bias")
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
  def model_fn(inp, tgt, mems, is_training):
    #### Set inputs
    features = {
        "inputs": inp,
        "target": tgt
    }

    if is_training:
      #### Get loss from inputs
      total_loss, new_mems, monitor_dict = model_function.get_lm_loss(
          features, mems, is_training)

      # number of parameters
      num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
      tf.logging.info('#params: {}'.format(num_params))

      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      grads_and_vars = list(zip(grads, all_vars))

      return loss, new_mems, grads_and_vars
    else:
      #### Get loss from inputs
      total_loss, logits, new_mems = model_function.get_lm_pred(
          features, mems, is_training)

      # number of parameters
      num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
      tf.logging.info('#params: {}'.format(num_params))

      return total_loss, logits, new_mems

  return model_fn


def single_core_graph(is_training, inp, tgt, mems):
  model_fn = get_model_fn()

  model_ret = model_fn(
      inp=inp,
      tgt=tgt,
      mems=mems,
      is_training=is_training)

  return model_ret


def evaluate(ps_device):
  ##### Get input function and model function
  eval_input_fn, eval_record_info = input_function.get_input_fn(
      tfrecord_dir=FLAGS.record_dir,
      split=FLAGS.eval_split,
      bsz_per_host=FLAGS.eval_batch_size,
      seq_len=FLAGS.seq_len,
      num_hosts=1,
      num_core_per_host=FLAGS.num_core_per_host,
      lower_case=FLAGS.lower_case,
      num_passes=1,
      use_bfloat16=False)

  num_batch = eval_record_info["num_batch"]
  if FLAGS.max_eval_batch > 0:
      num_batch = FLAGS.max_eval_batch
  tf.logging.info("num of batches {}".format(num_batch))

  ##### Create computational graph
  eval_set = eval_input_fn({
      "batch_size": FLAGS.eval_batch_size,
      "data_dir": FLAGS.data_dir})

  features = eval_set.make_one_shot_iterator().get_next()
  inputs_feed, target_feed = features["inputs"], features["target"]

  inputs = tf.split(inputs_feed, FLAGS.num_core_per_host, 0)
  target = tf.split(target_feed, FLAGS.num_core_per_host, 0)

  per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
  tower_mems, tower_losses, tower_new_mems, tower_logits = [], [], [], []

  for i in range(FLAGS.num_core_per_host):
    # with tf.device(assign_to_gpu(i, ps_device)), \
    #     tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

      mems_i = [tf.placeholder(tf.float32,
                    [per_core_bsz, FLAGS.mem_len, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]

      loss_i, logits_i, new_mems_i = single_core_graph(
          is_training=False,
          inp=inputs[i],
          tgt=target[i],
          mems=mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_logits.append(logits_i)
      tower_new_mems.append(new_mems_i)

  ## sum losses across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
  else:
    loss = tower_losses[0]

  ##### Evaluation loop
  tower_mems_np = [
      [np.zeros([per_core_bsz, FLAGS.mem_len, FLAGS.d_model], dtype=np.float32)
          for layer in range(FLAGS.n_layer)]
      for core in range(FLAGS.num_core_per_host)
  ]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path
    tf.logging.info("Evaluate {}".format(eval_ckpt_path))
    if eval_ckpt_path is not None:
      saver.restore(sess, eval_ckpt_path)

    total_loss, total_cnt = 0, 0
    format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
        len(str(num_batch)))

    if FLAGS.renormalize:
      # Initialize additional structures needed for renormalization
      tf.logging.info("Initialize renormalize vocab")
      renormalize_vocab = RenormalizeVocabPost(
          vocab_map_fn=os.path.join(FLAGS.vocab_dir, 'vocab_map.txt'),
          vocab_fn=os.path.join(FLAGS.vocab_dir, 'vocab.txt'),
          special=['<eos>']
      )
      renormalize_vocab.build_renormalize_vocab()
      tower_status_np = [
          renormalize_vocab.build_initial_status(per_core_bsz)
      ]

      # Different fetches
      fetches = [inputs, target, tower_logits, tower_new_mems]

      for step in range(num_batch):
        if step % (num_batch // 10) == 0:
          tf.logging.info(format_str.format(step, num_batch))

        feed_dict = {}
        for i in range(FLAGS.num_core_per_host):
          for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
            feed_dict[m] = m_np

        fetched = sess.run(fetches, feed_dict=feed_dict)
        inputs_np, target_np, tower_logits_np, tower_mems_np = fetched

        # get normalized loss
        for i, (inp, tgt, logits, status) in enumerate(zip(inputs_np,
                                                           target_np,
                                                           tower_logits_np,
                                                           tower_status_np)):
          loss_np, last_status = renormalize_vocab.get_eval_loss(
              inp,
              tgt,
              logits,
              status)
          total_loss += loss_np.sum()
          total_cnt += loss_np.size()
          tower_status_np[i] = last_status
    else:
      fetches = [loss, tower_new_mems, tf.size(target_feed)]

      for step in range(num_batch):
        if step % (num_batch // 10) == 0:
          tf.logging.info(format_str.format(step, num_batch))

        feed_dict = {}
        for i in range(FLAGS.num_core_per_host):
          for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
            feed_dict[m] = m_np

        fetched = sess.run(fetches, feed_dict=feed_dict)

        loss_np, tower_mems_np, cnt_np = fetched[:3]
        total_loss += loss_np * cnt_np
        total_cnt += cnt_np

    avg_loss = total_loss / total_cnt
    tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.get_tokenizer(
      tokenizer_type=FLAGS.tokenizer_type,
      paths=FLAGS.tokenizer_paths.split(","),
      do_lower_case=FLAGS.lower_case)
  data_utils.setup_special_ids(tokenizer)

  evaluate("/gpu:0")


if __name__ == "__main__":
  tf.app.run()
