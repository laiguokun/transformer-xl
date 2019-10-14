from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
try:
  import google3.experimental.users.zihangd.lm.data_utils as data_utils
  import google3.experimental.users.zihangd.lm.input_function as input_function
  import google3.experimental.users.zihangd.lm.tokenization as tokenization
except ImportError as e:
  import data_utils
  import input_function
  import tokenization


flags.DEFINE_string("record_info_dir", default="./proc_data/example",
                    help="Path to local directory containing *.tfrecord.")
flags.DEFINE_string("tokenizer_type", "sent_piece",
                    help="Type of the tokenizer.")
flags.DEFINE_string("tokenizer_paths", "",
                    help="Comma separated string.")
flags.DEFINE_integer("num_core_per_host", default=16,
                     help="number of cores per host.")
flags.DEFINE_integer("bsz_per_host", default=32, help="batch size per host.")
flags.DEFINE_integer("seq_len", default=512,
                     help="tgt len; 0 for not using it")
flags.DEFINE_bool("lower_case", default=False,
                  help="use lower_case inputs or not.")
flags.DEFINE_string("split", default="train",
                    help="Data split.")
flags.DEFINE_integer("num_passes", default=None,
                     help="Num of passes to use.")

FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv  # Unused

  tokenizer = tokenization.get_tokenizer(
      tokenizer_type=FLAGS.tokenizer_type,
      paths=FLAGS.tokenizer_paths.split(","),
      do_lower_case=FLAGS.lower_case)
  data_utils.setup_special_ids(tokenizer)

  input_fn, record_info_dict = input_function.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split=FLAGS.split,
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      num_hosts=1,
      num_core_per_host=FLAGS.num_core_per_host,
      lower_case=FLAGS.lower_case,
      num_passes=FLAGS.num_passes,
  )
  tf.logging.info(record_info_dict)

  bsz_per_core = FLAGS.bsz_per_host // FLAGS.num_core_per_host
  params = {
      "batch_size": bsz_per_core
  }

  dataset = input_fn(params)
  example = dataset.make_one_shot_iterator().get_next()

  with tf.Session() as sess:
    for i in range(10):
      example_np = sess.run(example)
      if i == 0:
        for key in example_np.keys():
          print(key, example_np[key].shape, len(example_np[key].shape))
          print(example_np[key][0].tolist())
          print(tokenizer.convert_ids_to_text(example_np[key][0].tolist()))
        print("=" * 80)

if __name__ == "__main__":
  tf.app.run()
