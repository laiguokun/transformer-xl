# Lint as: python3
"""Copy from tensor2tensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import multiprocessing

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS


def pack_dataset(dataset, length, keys=None, use_custom_ops=False):
  """Creates a 'packed' version of a dataset on-the-fly.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.
  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.
  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"source": [8, 7, 1, 0], "target":[4, 1, 0]}
  {"source": [2, 3, 4, 1], "target":[5, 6, 1]}
  The output example is:
  {
                "source": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
   "source_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
       "source_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "target": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "target_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "target_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the source and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".
  Args:
    dataset: a tf.data.Dataset
    length: an integer
    keys: a list of strings (e.g. ["source", "target"])
    use_custom_ops: use a custom c++ op not included in standard tf (faster)
  Returns:
    a tf.data.Dataset
  """
  shapes = dataset.output_shapes
  if keys is None:
    keys = shapes.keys()

  for k in keys:
    if k not in shapes:
      raise ValueError("Key %s not found in dataset.  Available keys are %s"
                       % (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError("Tensors to be packed must be one-dimensional.")

  if use_custom_ops:
    raise NotImplementedError
    # return _pack_with_custom_ops(dataset, keys, length)
  else:
    packer = SequenceDatasetPacker(length, spacing=0, queue_size=10)
    return packer(dataset, cycle_length=10, keys=keys)


#####################################
##### Tensorflow Implementation #####
#####################################
INDEX_DTYPE = tf.int32


class SequenceDatasetPacker(object):
  """Helper class for packing a dataset of sequences in an online fashon.

  The input sequence is expected to be a tuple of 1D Tensors which will be
  converted to a dataset which produces a dict of packed examples, example
  positions, and segment ids.
  If `window_size` or `cycle_length` is specified multiple packing operations
  will be performed in parallel to increase throughput. A value of None will
  select default parallelism parameters. If this dataset will be run on a TPU,
  specifying a cycle_length > 10 is recommended.
  """

  def __init__(self, packed_length=256, spacing=0, queue_size=10,
               chop_long_sequences=False):
    self._packed_length = packed_length
    self._spacing = spacing
    self._queue_size = queue_size
    self._chop_long_sequences = chop_long_sequences
    self._num_sequences = None
    self._token_dtype = None

  def __call__(self, dataset, **kwargs):
    if {"window_size", "cycle_length"}.intersection(kwargs):
      return self._concurrent_pack(dataset, **kwargs)
    return self._pack(dataset, **kwargs)

  def _concurrent_pack(self, dataset, window_size=None, cycle_length=None,
                       keys=None):
    """Selects sensible default parallelism parameters based for a task."""

    if window_size is None:
      # This is a heuristic to fill all of the queues 10 times, and should do a
      # reasonable job balancing parallelism (which benefits from lower window
      # size) with packing efficiency (which suffers from edge effects when the
      # window size is too low.)
      window_size = int(self._packed_length / 8 * self._queue_size * 10)

    if cycle_length is None:
      # Typically binning one stream will saturate about 3 cores.

      # Note on TPUs:
      # cycle_length should still be explicitly set when training on TPUs,
      # since the cpu count will be the local CPU count (which could be quite
      # small), wereas the transforms will actually run on the TPU host
      # controller which has a very robust CPU.
      cycle_length = max([int(multiprocessing.cpu_count() / 3), 1])
    return self._pack(dataset, window_size=window_size,
                      cycle_length=cycle_length, keys=keys)

  def _pack(self, dataset, window_size=None, cycle_length=None,
            deterministic=False, keys=None):
    """Main method for chaining together packing transformation steps."""
    (dataset, self._num_sequences, self._token_dtype, keys
    ) = self._standardize(dataset, keys)
    if window_size is None:
      dataset = self._scanning_pack(dataset)
    else:
      # Dataset.window splits nested Tensors.
      re_zip = lambda *x: tf.data.Dataset.zip(x)
      dataset = dataset.window(window_size).map(re_zip).interleave(
          self._scanning_pack, cycle_length=cycle_length,
          block_length=window_size,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

      if not deterministic:
        # Sloppy interleave offers a marginal performance improvement.
        options = tf.data.Options()
        options.experimental_deterministic = False
        dataset = dataset.with_options(options)

    dataset = dataset.map(
        self._finalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self._num_sequences, self._token_dtype = None, None

    if keys:
      def dict_pack(example):
        output = {}
        for i, key in enumerate(keys):
          output[key] = example["contents"][:, i]
          output[key + "_segmentation"] = example["segment"][:, i]
          output[key + "_position"] = example["position"][:, i]
        return output
      dataset = dataset.map(dict_pack)
    return dataset

  def _standardize(self, dataset, keys):
    """Force dataset structure into a tuple of Tensors."""
    shapes = tf.compat.v1.data.get_output_shapes(dataset)

    if isinstance(shapes, dict):
      keys = keys or tuple(shapes.keys())
      dataset = dataset.map(lambda x: tuple(x[k] for k in keys))
      shapes = tf.compat.v1.data.get_output_shapes(dataset)

    if not all(isinstance(i, tf.TensorShape) for i in shapes):
      # Internally this class expects tuples of Tensors, even for the degenerate
      # case of a single sequence.
      dataset = dataset.map(lambda x: (x,))
      shapes = tf.compat.v1.data.get_output_shapes(dataset)

    for s in shapes:
      if not s.is_compatible_with(tf.TensorShape([None])):
        raise ValueError("Tensors to be packed must be one-dimensional.")

    if not shapes:
      raise ValueError("Expected sequence dataset.")

    if self._chop_long_sequences and len(shapes) != 1:
      raise ValueError("chop_long_sequences expects a single sequence dataset.")

    token_types = tf.compat.v1.data.get_output_types(dataset)
    if len(set(token_types)) > 1:
      raise ValueError("Inconsistent dtypes: {}".format(token_types))

    return dataset, len(shapes), token_types[0], keys

  def _eviction_fn(self, _):
    return tuple(-tf.ones((self._packed_length,), dtype=self._token_dtype)
                 for _ in range(self._num_sequences))

  def _scan_initial_state(self):
    """Create TensorArrays and indices to track bin assignment.

    availability: TensorArray[queue_size, num_sequences]
      This represents the number of tokens available in the ith bin.
      See implementation note below.
    contents: TensorArray[queue_size, num_sequences * 2]
      This holds the actual contents of the packed strings as well as a bit
      mask indicating where sequences begin. It is stored in a flat vector and
      is accessed in offsets of packed_length.
    top_index: scalar [0, queue_size)
      Integer tensor indicating which index is the "top" bin. See implementation
      note below.
    IMPLEMENTATION_NOTE:
      The FFD algorithm periodically pops the topmost queue and pushes a new
      one to replace it. In order to replicate those semantics with a fixed size
      TensorArray, indexing operations are shifted by top_index. For example,
      instead of:
        `queue_available.read(i)`
      a read is instead performed as:
        `queue_available.read((i - top_index) % queue_size)`
      to account for the fact that the "ith" logical FFD queue is stored at
      position j. This means that the pop / push update can be performed by
      simply incrementing top_index. (And zeroing the old top_index position.)
    Returns:
      The state for the binning scan.
    """

    all_available = tf.ones((self._queue_size, self._num_sequences),
                            dtype=INDEX_DTYPE) * self._packed_length
    total_size = self._packed_length * self._queue_size
    total_size_range = tf.range(total_size, dtype=INDEX_DTYPE)
    empty = tf.zeros((total_size, self._num_sequences * 2),
                     dtype=self._token_dtype)

    availability = tf.TensorArray(
        dtype=INDEX_DTYPE, size=self._queue_size, dynamic_size=False,
        clear_after_read=False, element_shape=(self._num_sequences,)
        ).scatter(tf.range(self._queue_size, dtype=INDEX_DTYPE), all_available)

    contents = tf.TensorArray(
        dtype=self._token_dtype, size=total_size, dynamic_size=False,
        clear_after_read=False, element_shape=(self._num_sequences * 2,)
        ).scatter(total_size_range, empty)

    # Which index should be considered the "top" bucket for the purpose of
    # the first-fit descending algorithm.
    top_index = tf.zeros((), dtype=INDEX_DTYPE)

    return availability, contents, top_index

  def _scanning_pack(self, dataset):
    """Apply scan based pack to a dataset."""
    if self._chop_long_sequences:
      dataset = dataset.map(lambda x: (x[:self._packed_length],))
    else:
      dataset = dataset.filter(lambda *x: tf.reduce_max(  # pylint: disable=g-long-lambda
          tf.stack([tf.shape(i)[0] for i in x]), axis=0) <= self._packed_length)

    # In order to retrieve the sequences which are still in the queue when the
    # dataset is exhausted, we feed dummy sequences which are guaranteed to
    # displace the remaining elements.
    dataset = dataset.concatenate(
        tf.data.Dataset.range(self._queue_size).map(self._eviction_fn))

    initial_state = self._scan_initial_state()
    step_fn = functools.partial(
        tf.autograph.to_graph(_scan_step_fn), packed_length=self._packed_length,
        queue_size=self._queue_size, spacing=self._spacing,
        num_sequences=self._num_sequences, token_dtype=self._token_dtype)

    dataset = dataset.apply(tf.data.experimental.scan(initial_state, step_fn))

    is_real_sample = lambda valid_sample, _: valid_sample
    return dataset.filter(is_real_sample)

  def _compute_auxiliary_structure(self, contents_and_mask):
    """Compute segment and position metadata."""
    contents = contents_and_mask[:, :self._num_sequences]
    start_mask = tf.cast(contents_and_mask[:, self._num_sequences:],
                         dtype=INDEX_DTYPE)

    segment = tf.cumsum(start_mask, axis=0)
    uniform_count = tf.ones_like(segment[:, 0])
    position = []
    for i in range(self._num_sequences):
      segment_slice = segment[:, i]
      counts = tf.math.segment_sum(uniform_count, segment[:, i])
      position.append(tf.range(self._packed_length) -  tf.cumsum(
          tf.gather(counts, segment_slice - 1) * start_mask[:, i]))
    position = tf.concat([i[:, tf.newaxis] for i in position], axis=1)

    # Correct for padding tokens.
    pad_mask = tf.cast(tf.not_equal(contents, 0), dtype=INDEX_DTYPE)
    segment *= pad_mask
    position *= pad_mask

    return segment, position

  def _finalize(self, _, contents):
    """Structure output and compute segment and position metadata."""

    # The output shape information is lost during the filter; however we can
    # guarantee the shape. (That's the point of this exercise, after all!)
    contents.set_shape((self._packed_length, self._num_sequences * 2))

    # Both the dummy branch of the scan step function and the eviction dataset
    # use vectors of minus one. The cost of this check is negligible and the
    # leakage of such dummy sequences would be difficult to debug downstream.
    check_leaks = tf.assert_none_equal(contents, -tf.ones_like(contents))
    with tf.control_dependencies([check_leaks]):
      contents = tf.identity(contents)

    segment, position = self._compute_auxiliary_structure(contents)
    return {"contents": contents[:, :self._num_sequences],
            "segment": segment, "position": position}


def _scan_step_fn(state, example, packed_length, queue_size, spacing,
                  num_sequences, token_dtype):  # pylint: disable=g-doc-args
  """Transform function used by tf.data.experimental.scan to process an example.

  This is written as a stateless function rather than a class method because we
  trace it with AutoGraph (in order to simplify the conditional), and this way
  we don't have to worry about handling re-tracing semantics.
  Args:
    See the SequenceDatasetPacker class.
  Returns:
    The updated queue state, and either a packed example or a dummy sequence
    which will be filtered out downstream.
  """

  # Convert TensorArray tuples to lists since we'll need to replace them.
  availability, contents, top_index = state

  lengths = tf.concat([tf.shape(i) for i in example], axis=0)
  start_availability = availability.stack()
  can_fit = tf.reduce_all(tf.greater_equal(start_availability, lengths), axis=1)
  any_can_fit = tf.reduce_any(can_fit, axis=0)

  # AutoGraph will convert this block to a tf.cond
  if any_can_fit:
    # This indicates where in the FFD queue rotation a given index sits
    shifted_range = (
        tf.range(queue_size, dtype=INDEX_DTYPE) - top_index) % queue_size

    # Mark any indices which cannot accommodate the current example.
    exclusion_mask = tf.cast(tf.logical_not(can_fit), INDEX_DTYPE) * queue_size

    # Index in [0, queue_size) in which to place the sample. Note, this index
    # is the position in the actual TensorArray, not the index of the FFD queue.
    queue_index = (tf.reduce_min(shifted_range + exclusion_mask) +
                   top_index) % queue_size

    # NOTE(taylorrobie): We emit a non-empty Tensor for downstream checks.
    output_contents = -tf.ones((1, num_sequences), dtype=token_dtype)

  else:
    index_range = top_index * packed_length + tf.range(packed_length)
    output_contents = contents.gather(index_range)

    # Reset the queue state.
    availability = availability.write(
        top_index, packed_length * tf.ones((num_sequences,), dtype=INDEX_DTYPE))
    empty_contents = tf.zeros((packed_length, num_sequences * 2),
                              dtype=token_dtype)
    contents = contents.scatter(index_range, empty_contents)

    queue_index = top_index
    top_index = (top_index + 1) % queue_size

  pre_assign_availability = availability.read(queue_index)
  space_left = pre_assign_availability - lengths - spacing
  availability = availability.write(queue_index, space_left)

  # ============================================================================
  # == Update contents =========================================================
  # ============================================================================
  # Consider the following case for a seq-to-seq packing:
  #   (padding is represented as underscores)
  #
  #   Queue starting state:
  #     [1, 3, 2, 4, 6, 1, _, _, _, _, _, ...]
  #     [5, 9, _, _, _, _, _, _, _, _, _, ...]
  #
  #   Examples:
  #     [4, 2, 4], [3]
  #
  #   Desired new queue state:
  #     [1, 3, 2, 4, 6, 1, _, _, 4, 2, 4, _, _, ...]
  #     [5, 9, _, _, 3, _, _, _, _, _, _, _, _, ...]
  #
  # This could be acomplished by creating a TensorArray for each of the two
  # sequences, and scattering into the respective arrays. However TensorArray
  # writes are extremely expensive relative to other operations. So instead we
  # store the contents in a single TensorArray of shape (packed_length, 2), and
  # we pad and concatenate the examples such that they can be added in a single
  # assign:
  #
  #              [_, _, _, _, 4, 2, 4]
  #              [3, _, _, _, _, _, _]
  #                        +
  #  [1, 3, 2, 4, 6, 1, _, _, _, _, _, ...]
  #  [5, 9, _, _, _, _, _, _, _, _, _, ...]
  #
  # And in practice, the extra work of padding is neglidgable compared to
  # the gain from vectorizing the TensorArray assign. We also store a bit mask
  # denoting where sequences start which is used to compute segment and
  # position metadata:
  #
  #              [_, _, _, _, 1, _, _]
  #              [1, _, _, _, _, _, _]
  #                        +
  #  [1, _, _, _, _, _, _, _, _, _, _, ...]
  #  [1, _, _, _, _, _, _, _, _, _, _, ...]
  #
  # Both the contents and the mask are concatenated in the same TensorArray
  # for performance.

  start_index = packed_length - pre_assign_availability
  end_index = start_index + lengths
  leftmost = tf.reduce_min(start_index, axis=0)
  rightmost = tf.reduce_max(end_index, axis=0)
  delta = rightmost - leftmost
  pad_indices = [tf.stack((start_index[i] - leftmost, rightmost - end_index[i]))
                 for i in range(num_sequences)]

  padded_examples = [tf.pad(ex, padding[tf.newaxis, :])
                     for ex, padding in zip(example, pad_indices)]
  padded_examples = tf.transpose(tf.stack(padded_examples))
  mask_update = tf.one_hot(start_index - leftmost, delta,
                           dtype=contents.dtype, axis=0)

  content_update = tf.concat([padded_examples, mask_update], axis=1)

  index_range = (queue_index * packed_length +  # Offset into the right section.
                 tf.range(delta, dtype=INDEX_DTYPE) + leftmost)
  contents = contents.scatter(index_range, contents.gather(index_range) +
                              content_update)

  state = (availability, contents, top_index)
  return state, (tf.logical_not(any_can_fit), output_contents)

