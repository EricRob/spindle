# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing EEG text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import pdb

def _read_data(filename):
  #with tf.gfile.GFile(filename, "r") as f:
  #    return tf.string_to_number(f.read().split())
  with open(filename) as f:
    numbers_str = f.read().split()
    numbers_float = [float(x) for x in numbers_str]
  return numbers_float


def spindle_raw_data(data_path=None):
  """Load EEG raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  train_list = ['excerpt2', 'excerpt4', 'excerpt5', 'excerpt6', 'Augment_train_400_2c']
  #train_list = ['Augment']
  valid_list = ['excerpt7']
  test_list = ['excerpt8']
  train_data = []
  train_label = []
  valid_data = []
  valid_label = []
  test_data = []
  test_label = []

  for x in train_list:
    train_path = os.path.join(data_path, x + ".txt")
    train_data += _read_data(train_path)
    train_label_path = os.path.join(data_path, x + "_labels.txt")
    train_label += _read_data(train_label_path)

  for x in valid_list:
    valid_path = os.path.join(data_path, x + ".txt")
    valid_data += _read_data(valid_path)
    valid_label_path = os.path.join(data_path, x + "_labels.txt")
    valid_label += _read_data(valid_label_path)

  for x in test_list:
    test_path = os.path.join(data_path, x + ".txt")
    test_data += _read_data(test_path)
    test_label_path = os.path.join(data_path, x + "_labels.txt")
    test_label += _read_data(test_label_path)

  return train_data, valid_data, test_data, train_label, valid_label, test_label


def spindle_producer(raw_data, raw_labels, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "SpindleProducer", [raw_data, raw_labels, batch_size, num_steps]):

    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)
    raw_labels = tf.convert_to_tensor(raw_labels, name="raw_labels", dtype=tf.float32)
    raw_labels = tf.to_int32(raw_labels)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    labels = tf.reshape(raw_labels[0 : batch_size * batch_len],
                      [batch_size,batch_len])
    epoch_size = (batch_len - 1) // num_steps

    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")

    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    #for _ in range(4):
    i = tf.train.range_input_producer(10, shuffle=True).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(labels, [0, i * num_steps],
                        [batch_size, (i + 1) * num_steps])
    y.set_shape([batch_size, num_steps])
    y = tf.cast(tf.greater(tf.reduce_sum(y,1),49),tf.int32)
    #pdb.set_trace()
    return x, y