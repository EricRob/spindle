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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_rnn_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import SpindleDataset_noplace as SpindleDataset
import util
import pdb
from tensorflow.python.client import device_lib
from math import sqrt
from scipy.signal import butter, lfilter
from termcolor import cprint

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", '/data/spindle/SleepSpindleData4RNN',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
flags.DEFINE_float("lr", 1e-4,
                   "Learning rate for Traing.")
flags.DEFINE_float("drop", 0.5,
                   "Keep dropout probalitity." )
flags.DEFINE_integer("test", 0,
                    "test mode if 1")
flags.DEFINE_integer("target_f", 200, "sampled data frequency")
flags.DEFINE_string("model_path","model","Path to stored model for testing")
flags.DEFINE_string("test_sub", '01', 'Subject for model testing')
flags.DEFINE_integer("upsampled", 0, "original frequency if upsampled")
flags.DEFINE_integer("fold", 1, "fold number for choosing training subjects (five-fold validation). not relevant for testing.")

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

def get_data_size(filename):
  with open(filename) as f:
    return sum(1 for _ in f)


def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # reshape kenel to 4-D tensor
  kernel = tf.reshape(kernel, tf.stack([kernel.get_shape()[1], kernel.get_shape()[0], 1, kernel.get_shape()[2]]))
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad,pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1]+ 2* pad
  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))


  # scaling to [0, 255] is not necessary for tensorboard
  return x

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class SpindleInput(object):
  """The input data."""

  def __init__(self, config, data, label, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    #pdb.set_trace()
    self.input_data, self.targets = reader.spindle_producer(
        data, label, batch_size, num_steps, name=name)

class SpindleInputDatasetAPI(object):
  """The input data using Dataset API"""

  def __init__(self, config, name=None):
      self.batch_size = batch_size = config.batch_size
      self.num_steps = num_steps = config.num_steps
      self.target_f = target_f = FLAGS.target_f
      filename = []
      #self.subject = subject =  [1] #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
      subject = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
      for remove in np.arange(4*(FLAGS.fold-1) + 1,4*FLAGS.fold + 1):
        if remove in subject:
          subject.remove(remove)

      self.subject = subject

      self.test_sub = test_sub = [FLAGS.test_sub]
      for k in self.subject:
        filename.append('Augment_' + name + '_MASS_sub' + str(k) + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_mf.txt')
      #filename = ['_Dream_50_synth2.txt','_MASS_sub1_50_100%_f200.txt']
      if name == 'train':
        if config.test_mode:
            self.epoch_size = 1000
        else:
            self.epoch_size = (sum([get_data_size(os.path.join(FLAGS.data_path, x)) for x in filename]) // batch_size) // num_steps

      elif name == 'valid':
          if config.test_mode:
              self.epoch_size = 1000
          else:
              self.epoch_size = (sum([get_data_size(os.path.join(FLAGS.data_path, x)) for x in filename]) // batch_size) // num_steps
      elif name == 'test':
          if config.test_mode:
              if FLAGS.upsampled:
                  self.epoch_size = (get_data_size(os.path.join(FLAGS.data_path, 'upsampled_test_01-02-00' + test_sub[0] + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_from_' + str(FLAGS.upsampled) + '_mf.txt')) // batch_size)
              else:
                  self.epoch_size = (get_data_size(os.path.join(FLAGS.data_path, 'test_01-02-00' + test_sub[0] + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_mf.txt')) // batch_size)
          else:
              self.epoch_size = (sum([get_data_size(os.path.join(FLAGS.data_path, x)) for x in filename]) // batch_size) // num_steps  #self.epoch_size = (get_data_size('SleepSpindleData4RNN/test_MrOS_' + test_sub[0] + '_C3_50_100%_f200_mf.txt') // batch_size)
      else:
        raise ValueError('Invalid name "%s"' % name)
      #pdb.set_trace()
      spindle_dataset = SpindleDataset.SpindleDatasetBatcher(FLAGS.data_path, name, num_steps, subject, test_sub, target_f, FLAGS.upsampled)
      data_batch, label_batch = spindle_dataset.spindle_producer_datasetAPI(batch_size)

      self.input_data = tf.unstack(data_batch, num=batch_size, axis=0)
      self.targets = tf.unstack(label_batch, num=batch_size, axis=0)

class SpindleModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    output_size = 2

    #with tf.device("/gpu:0"):
      #embedding = tf.get_variable(
      #    "embedding", [1, size], dtype=data_type())
      #inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      #inputs = tf.reshape(input_.input_data,[-1, 1])
      #inputs = tf.matmul(inputs, embedding)

    # Add CNN layer
    #pdb.set_trace()
    input_.input_data = tf.stack(input_.input_data)
    ratio = tf.concat([tf.reshape(input_.input_data[:,int(self.num_steps/2),2],[self.batch_size, -1]), input_.input_data[:,int(self.num_steps/2),4:]],1)
    #input_.input_data = tf.concat([tf.reshape(input_.input_data[:,:,0],[self.batch_size, self.num_steps, -1]), tf.reshape(input_.input_data[:,:,3],[self.batch_size, self.num_steps, -1])],2)
    inputs_ =tf.reshape(input_.input_data[:,:,0],[self.batch_size, self.num_steps, -1])
    inputs_2 =tf.reshape(input_.input_data[:,:,3],[self.batch_size, self.num_steps, -1])
    # data shape is "[batch, in_height, in_width, in_channels] so-called NHWC"
    #inputs_ = tf.reshape(input_.input_data, [self.batch_size, self.num_steps, -1])

    # normalization across sequence
    #inputs_ = tf.layers.batch_normalization(inputs_, axis=1, momentum=0.1,
    #                                         training=True,epsilon=1e-5,
    #                                         name="norm")
    #inputs_ = tf.nn.l2_normalize(inputs_, dim=1, epsilon=1e-12, name="norm")
    #pdb.set_trace()
    #cnn_output = self._add_conv_layers(inputs_, config, is_training)

    #batch = tf.Variable(tf.zeros([self.batch_size, self.num_steps]),tf.float32)
    #batch = tf.assign(batch, input_.input_data)
    #fir = tf.constant([0.0421, 0.4728, 0.4728, 0.0421])
    #fir = tf.reshape(fir, [4, 1, 1])
    #batch_bp = tf.reshape(batch, [self.batch_size, self.num_steps, 1])
    #batch_bp = tf.nn.conv1d(batch_bp, fir, stride=1, padding="SAME")
    #batch_bp = tf.squeeze(batch_bp)
    #pdb.set_trace()
    cnn_output1 = self._add_conv_layers(inputs_, config, is_training, 0)
    cnn_output2 = self._add_conv_layers(inputs_2, config, is_training, 1)
    #cnn_output = tf.concat([cnn_output1,cnn_output2], 2)

    inputs = tf.reshape(cnn_output1, [self.batch_size, self.num_steps, -1])
    inputs2 = tf.reshape(cnn_output2, [self.batch_size, self.num_steps, -1])
    #pdb.set_trace()
    # add CNN attention module
    #inputs = tf.transpose(inputs, perm=[0,2,1])
    #inputs2 = tf.transpose(inputs2, perm=[0,2,1])
    #c_a = tf.layers.dense(inputs, self.num_steps, tf.nn.elu)
    #c_a = tf.layers.dense(inputs, self.num_steps, tf.nn.elu)
    #c_a2 = tf.layers.dense(inputs2, self.num_steps, tf.nn.elu)
    #c_a2 = tf.layers.dense(inputs2, self.num_steps, tf.nn.elu)
    #inputs = tf.multiply(inputs, c_a)
    #inputs2 = tf.multiply(inputs2, c_a2)
    #inputs = tf.transpose(inputs, perm=[0,2,1])
    #inputs2 = tf.transpose(inputs2, perm=[0,2,1])

    if config.keep_prob < 1:
      inputs = tf.layers.dropout(inputs, 1-config.keep_prob, training=is_training)
      inputs2 = tf.layers.dropout(inputs, 1-config.keep_prob, training=is_training)

    output, state = self._build_rnn_graph(inputs, config, is_training)
    output2, state2 = self._build_rnn_graph(inputs2, config, is_training, "rnn2")

    #pdb.set_trace()
    # add RNN attention module
    #output = tf.transpose(output, perm=[0,2,1])
    #output2 = tf.transpose(output2, perm=[0,2,1])
    #r_a = tf.layers.dense(output, self.num_steps, tf.nn.elu)
    #r_a = tf.layers.dense(output, self.num_steps, tf.nn.elu)
    #r_a2 = tf.layers.dense(output2, self.num_steps, tf.nn.elu)
    #r_a2 = tf.layers.dense(output2, self.num_steps, tf.nn.elu)
    #output = tf.multiply(output, r_a)
    #output2 = tf.multiply(output2, r_a2)
    #output = tf.transpose(output, perm=[0,2,1])
    #output2 = tf.transpose(output2, perm=[0,2,1])

    #output = tf.reshape(output, [self.batch_size, self.num_steps, size])
    output = tf.reshape(output, [self.batch_size, -1])
    output2 = tf.reshape(output2, [self.batch_size, -1])
    #output = tf.transpose(output, perm=[0, 2, 1])

    output = tf.concat([output, output2, tf.reshape(ratio, [self.batch_size, -1])], 1)
    fc1 = tf.layers.dense(output, 50, tf.nn.elu)

    #stfts = tf.abs(tf.contrib.signal.stft(batch, frame_length=self.num_steps, frame_step=1, fft_length=self.num_steps))
    #power = tf.real(stfts*tf.conj(stfts))
    #log_offset = 1e-6
    #log_power = tf.squeeze(tf.log(power+log_offset))
    #low_f = tf.reduce_sum(log_power[:,1:2], 1, keep_dims=True)
    #s_f = tf.reduce_sum(log_power[:,3:4], 1, keep_dims=True)

    #mean, std = tf.nn.moments(batch,axes=1)
    #fc1 = tf.concat([fc1, tf.reshape(std,[self.batch_size, -1])], 1)
    #fc1 = tf.concat([fc1, batch_bp], 1)
    #fc2 = tf.layers.dense(fc1, 1, tf.nn.elu)
    #fc2 = tf.concat([fc2, low_f/s_f], 1)
    #pdb.set_trace()
    #fc2 = tf.layers.dense(tf.reshape(cnn_output, [self.batch_size, -1]), output_size)
    fc2 = tf.layers.dense(fc1, output_size)

    #softmax_w = tf.get_variable(
    #    "softmax_w", [size, output_size], dtype=data_type())
    #softmax_b = tf.get_variable("softmax_b", [output_size], dtype=data_type())

    #logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

    # Reshape logits to be a 3-D tensor for sequence loss
    #logits = tf.reshape(logits, [self.batch_size, self.num_steps, output_size])
    #logits = logits[:,self.num_steps-1,:]
    logits = fc2
    logits_scaled = tf.nn.softmax(logits)
    #pdb.set_trace()
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = input_.targets,logits = logits)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat([tf.reshape(1-tf.stack(input_.targets),[-1,1]),tf.reshape(tf.stack(input_.targets),[-1,1])],1), logits = logits)

    value, indice = tf.nn.top_k(logits_scaled, 1)

    # Update the cost
    batch = tf.Variable(tf.zeros([self.batch_size, self.num_steps]),tf.float32)
    #labels = tf.Variable(tf.to_int32(tf.ones([self.batch_size])))
    labels = tf.Variable(tf.to_float(tf.ones([self.batch_size])))
    self._labels = tf.assign(labels, tf.stack(input_.targets))
    self._input_data = tf.assign(batch, tf.stack(inputs_)[:,:,0])
    self._cost = tf.reduce_mean(loss)
    self._final_state = state
    #self._output = tf.to_int32(result[3])
    self._output = indice
    self._prob = logits_scaled[:,1]
    #self._output = eval_correct
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    #tvars = tf.trainable_variables()
    #grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
    #                                  config.max_grad_norm)
    #optimizer = tf.train.GradientDescentOptimizer(self._lr)
    optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-4)
    #self._train_op = optimizer.apply_gradients(
    #    zip(grads, tvars),
    #    global_step=tf.train.get_or_create_global_step())
    self._train_op = optimizer.minimize(self._cost, global_step=tf.train.get_or_create_global_step())
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _add_conv_layers(self, inputs, config, is_training, j=0):
    """ Adds convolution layers"""
    convolved = inputs
    #filter_size=[17,11,9,7,5]
    filter_size=[7,7,7,7,7]
    for i in range(config.num_cnn_layers):
      with tf.variable_scope("conv%s" % str(j+1)):
        convolved_input = convolved
        # Add dropout layer if enabled and not first convolution layer.
        if i > 0 and config.keep_prob < 1:
          convolved_input= tf.layers.dropout(
                convolved_input,
                rate=1-config.keep_prob,
                training=is_training)

          #convolved_input = tf.layers.batch_normalization(convolved_input, axis=-1, momentum=0.1,
          #                                  training=True,epsilon=1e-5,
          #                                  name="norm%s" % str(i+1))
        conv = tf.layers.conv1d(
              convolved_input,
              filters = 40,
              kernel_size = filter_size[i],
              activation = tf.nn.elu,
              strides = 1,
              padding = "same",
              name = "conv%s" % str(i+1))

        if i==0 and is_training:
          #pdb.set_trace()
          kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model/conv1/conv1/kernel')[0]
          grid = put_kernels_on_grid(kernel)
          tf.summary.image('Model/conv1/conv1/kernel', grid, max_outputs=1)
        #norm = tf.layers.batch_normalization(conv, axis=1, momentum=0.1,
        #                                     training=is_training,epsilon=1e-5,
        #                                     name="norm%s" % str(i+1))
        pool = tf.layers.max_pooling1d(conv, pool_size=5, strides=1, padding="same",
                                       name="pool%s" % str(i+1))
        convolved = pool
    return convolved



  def _build_rnn_graph(self, inputs, config, is_training, scope="rnn"):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training, scope)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_rnn_layers=config.num_rnn_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_rnn_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_rnn_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
              config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training, scope):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_rnn_layers)], state_is_tuple=True)

    cell_fw = [make_cell() for _ in range(config.num_rnn_layers)]
    cell_bw = [make_cell() for _ in range(config.num_rnn_layers)]

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                           initial_state=self._initial_state,
                           scope = scope)
    #outputs, state, output_state_bw = tf.contrib.rnn.stack_bidirectional_rnn(
    #        cells_fw=cell_fw,
    #        cells_bw=cell_bw,
    #        inputs=inputs,
    #        dtype=tf.float32,
    #        scope="rnn_classification")
    #outputs = []
    #with tf.variable_scope("RNN"):
    #  for time_step in range(self.num_steps):
    #    if time_step > 0: tf.get_variable_scope().reuse_variables()
    #    (cell_output, state) = cell(inputs[:, time_step, :], state)
    #    outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, self.num_steps, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    tf.add_to_collection(util.with_prefix(self._name, "output"),self._output)
    tf.add_to_collection(util.with_prefix(self._name, "prob"),self._prob)
    tf.add_to_collection(util.with_prefix(self._name, "labels"),self._labels)
    tf.add_to_collection(util.with_prefix(self._name, "input_data"),self._input_data)
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    self._output = tf.get_collection_ref(util.with_prefix(self._name, "output"))
    self._prob = tf.get_collection_ref(util.with_prefix(self._name, "prob"))
    self._labels = tf.get_collection_ref(util.with_prefix(self._name, "labels"))
    self._input_data = tf.get_collection_ref(util.with_prefix(self._name, "input_data"))
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def output(self):
    return self._output

  @property
  def prob(self):
    return self._prob
  @property
  def labels(self):
    return self._labels

  @property
  def input_data(self):
    return self._input_data

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.0
  max_grad_norm = 5
  num_cnn_layers = 5
  num_rnn_layers = 1
  num_steps = 50
  hidden_size = 100
  max_epoch = 10
  max_max_epoch = 30
  keep_prob = 0.5
  lr_decay = 1 #/ 1.05
  batch_size = 500 #20
  rnn_mode = BLOCK
  test_mode = 0

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.0001
  max_grad_norm = 5
  num_cnn_layers = 5
  num_rnn_layers = 1
  num_steps = 50
  hidden_size = 200
  max_epoch = 5
  max_max_epoch = 50
  keep_prob = 0.5
  lr_decay = 1 #/1.15
  batch_size = 200
  rnn_mode = BLOCK
  test_mode = 0

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_rnn_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_rnn_layers = 1
  num_steps = 50
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  predictions=[]
  probability=[]
  #correct_num = 0
  state = session.run(model.initial_state)
  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "output": model.output,
      "prob": model.prob,
      "labels": model.labels,
      "input_data": model.input_data
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  result = np.zeros(4)
  # epoch_size = num_examples / batch_size
  #pdb.set_trace()
  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    #vals = session.run(fetches, feed_dict)
    vals = session.run(fetches)
    cost = vals["cost"]
    state = vals["final_state"]
    output = vals["output"]
    prob = vals["prob"]
    labels = vals["labels"]
    input_data = vals["input_data"]
    costs += cost
    #correct_num += sum(output[0])
    iters += model.input.num_steps

    output = np.squeeze(output)
    prob = np.squeeze(prob)
    labels = np.squeeze(labels)
    predictions = np.append(predictions,output)
    probability = np.append(probability,prob)
    #if step==0:
    #pdb.set_trace()
    if np.size(labels) == 1:
      if labels >= 0.5:
        if output >= 0.5: result[0] += 1  # TP
        else: result[1] += 1  # FN
      else:
        if output < 0.5: result[2] += 1  # TN
        else: result[3] += 1  # FP
    else:
      for i in range(np.size(labels)):
        if labels[i] >= 0.5:
          if output[i] >=0.5:
            result[0] += 1  # TP
          else:
            result[1] += 1  # FN
        else:
          if output[i] < 0.5:
            result[2] += 1  # TN
          else:
            result[3] += 1  # FP

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f loss: %.3f speed: %.0f batches per secs" %
            (step * 1.0 / model.input.epoch_size, costs / (step + 1.0),
             model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))
    start_time = time.time()
  #print("Accuracy : %.3f  Epoch Size : %d " % (correct_num * 1.0 / model.input.epoch_size / model.input.batch_size, model.input.epoch_size))
  print("Sensitivity : %.3f  Specificity : %.3f  False Discovery Rate : %.3f False Positive Rate : %.3f Accuracy : %.3f Postive cases : %d  Negative cases : %d" % (result[0] * 1.0 / (result[0] + result[1]),
               result[2] * 1.0 / (result[2] + result[3]),
 	       result[3] * 1.0 / (result[0] + result[3]),
 	       result[3] * 1.0 / (result[2] + result[3]),
 	       (result[0] + result[2]) * 1.0 / np.sum(result),
               result[0] + result[1],
               result[2] + result[3]))
  time_cost = 1000*(time.time() - start_time)*1.0/(model.input.epoch_size * model.input.batch_size)
  print("Time Cost per example %.3f (ms)" % time_cost)
  os.makedirs("probability", exist_ok=True)
  prob_file_name = os.path.join("probability","probability_sub" + str(model.input.test_sub[0]) + "_" + str(FLAGS.target_f) + "Hz.txt")
  #pdb.set_trace()
  if FLAGS.upsampled:
      prob_file_name = os.path.join("probability", str(FLAGS.target_f) + "_from_" + str(FLAGS.upsampled) + "_" + "probability_sub" + str(model.input.test_sub[0]) + ".txt")
  np.savetxt(prob_file_name, probability.astype(float), fmt="%f")
  #np.savetxt("probability_MrOS_18_C3.txt", probability.astype(float), fmt="%f")
  return costs / model.input.epoch_size


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  if FLAGS.lr:
    config.learning_rate = FLAGS.lr
  if FLAGS.drop:
    config.keep_prob = FLAGS.drop
  if FLAGS.test:
    config.test_mode = FLAGS.test
  return config


def main(_):
  stdout_backup = sys.stdout
  #log_file = open("message.log", "a")
  #sys.stdout = log_file
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  #raw_data = reader.spindle_raw_data(FLAGS.data_path)
  #train_data, valid_data, test_data, train_label, valid_label, test_label = raw_data
  target_f = FLAGS.target_f

  config = get_config()
  config.num_steps = num_steps = int((250*0.001) / (1 / target_f))

  eval_config = get_config()
  eval_config.num_steps = num_steps
  #eval_config.batch_size = 20
  #eval_config.num_steps = 50

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):

      #train_input = SpindleInput(config=config, data=train_data, label=train_label, name="TrainInput")
      with tf.device('/cpu:0'):
        train_input = SpindleInputDatasetAPI(config=config, name="train")
      #pdb.set_trace()

      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = SpindleModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):

      #valid_input = SpindleInput(config=config, data=valid_data, label=valid_label, name="ValidInput")
      valid_input = SpindleInputDatasetAPI(config=config, name="valid")

      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = SpindleModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):

      #test_input = SpindleInput(config=eval_config, data=test_data, label=test_label, name="TestInput")
      test_input = SpindleInputDatasetAPI(config=eval_config, name="test")

      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = SpindleModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():

    tf.train.import_meta_graph(metagraph)

    for model in models.values():
      model.import_ops()
    
    if not FLAGS.test:
      os.makedirs(FLAGS.save_path, exist_ok=True)
    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=600)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement, gpu_options=gpu_options)

    #total_parameters = 0
    #for variable in tf.trainable_variables():
    #  # shape is an array of tf.Dimension
    #  shape = variable.get_shape()
    #  print(shape)
    #  print(len(shape))
    #  variable_parameters = 1
    #  for dim in shape:
    #    print(dim)
    #   variable_parameters *= dim.value
    #    print(variable_parameters)
    #    total_parameters += variable_parameters
    #print(total_parameters)
    #pdb.set_trace()
    print("=============================================New Test !!!!!==================================================")
    print(" Learning rate = %.6f   Keep Probability = %.2f " % (config.learning_rate, config.keep_prob))

    with sv.managed_session(config=config_proto) as session:
      gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
      if FLAGS.num_gpus > len(gpus):
      	raise ValueError(
      		"Your machine has only %d gpus "
        	"which is less than the requested --num_gpus=%d."
        	% (len(gpus), FLAGS.num_gpus))
      #pdb.set_trace()
      #sv.saver.restore(session, tf.train.latest_checkpoint(os.path.join(FLAGS.data_path, "../saved_model/")))
      if config.test_mode == 0:
        training_loss = []
        # sv.saver.restore(session, tf.train.latest_checkpoint(os.path.join(FLAGS.data_path, "../../Cross_Valid/1/")))
        for i in range(config.max_max_epoch):

          lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          m.assign_lr(session, config.learning_rate * lr_decay)
          cprint("Fold " + str(FLAGS.fold) + ", " + str(FLAGS.target_f) + "Hz", 'green' )
          if FLAGS.upsampled:
            cprint("Upsampled from " + str(FLAGS.upsampled) + "Hz") 

          print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
          train_loss = run_epoch(session, m, eval_op=m.train_op,
                                         verbose=True)
          print("Epoch: %d Train Loss: %.3f" % (i + 1, train_loss))
          valid_loss = run_epoch(session, mvalid, verbose=False)
          print("Epoch: %d" % (i+1))
          cprint("Valid Loss: %.3f" % (valid_loss), 'yellow')
          training_loss.append(train_loss)

        #fig=plt.figure()
        #plt.plot(training_loss)
        #fig.show()
          #test_loss = run_epoch(session, mtest, verbose=False)
          #print("Test Loss: %.3f" % test_loss)

        #variables_names =[v.name for v in tf.trainable_variables()]
        #value=session.run('Model/conv1/conv1/kernel:0')
        #pdb.set_trace()
        if FLAGS.save_path:
          print("Saving model to %s." % FLAGS.save_path)
          sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

      else:
        sv.saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_path))
        #sv.saver.restore(session, tf.train.latest_checkpoint(os.path.join(FLAGS.data_path, "../../Cross_Valid/1")))
        #pdb.set_trace()
        #variables_names =[v.name for v in tf.trainable_variables()]
        #value=session.run('Model/conv1/conv1/kernel:0')
        #pdb.set_trace()
        test_loss = run_epoch(session, mtest, verbose=False)
        print("Test Loss: %.3f" % test_loss)
  #log_file.close()
  sys.stdout = stdout_backup
if __name__ == "__main__":
  tf.app.run()

