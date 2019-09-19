# Lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", "data/example/*.txt",
                    help="Input file path.")


def main(argv):
  del argv

  title_pattern = re.compile("^ = [^=]+ = $")
  with tf.io.gfile.GFile(FLAGS.input_path, "r") as f:
    lines = [line.strip("\n") for line in f]
  tf.logging.info("Total number of lines: %d", len(lines))

  splits = []
  for idx, line in enumerate(lines):
    ret = title_pattern.match(line)
    if (ret is not None and not lines[idx-1].strip() and
        not lines[idx+1].strip()):
      splits.append(idx - 1)
  splits.append(len(lines))
  num_doc = len(splits) - 1
  tf.logging.info("Total number of docs: %d", num_doc)

  for idx, (beg, end) in enumerate(zip(splits[:-1], splits[1:])):
    out_path = "{}-{:05d}-of-{:05d}".format(FLAGS.input_path, idx, num_doc)
    tf.logging.info("Writing to doc %d: %s", idx, out_path)
    with tf.io.gfile.GFile(out_path, "w") as f:
      for line in lines[beg:end]:
        f.write(line + "\n")


if __name__ == "__main__":
  app.run(main)
