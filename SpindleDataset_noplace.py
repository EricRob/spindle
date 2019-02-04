from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import pdb
import numpy as np

class SpindleDatasetBatcher(object):
  """Spindle dataset"""

  def __init__(self, data_dir, subset='train', num_steps=1, subject=[1], test_sub=['01'], target_f=200, upsampled=0):
    self.data_dir = data_dir
    self.subset = subset
    self.num_steps = num_steps
    self.num_feats = 23
    self.subject = subject
    self.test_sub = test_sub
    self.target_f = target_f
    self.upsampled = upsampled

  def get_filenames(self, upsampled=0):
    if self.subset in ['train', 'valid']:
      filename = []
      for k in self.subject:
        filename.append(os.path.join(self.data_dir, self.subset + '_MASS_sub' + str(k) + '_' + str(self.num_steps) + '_100%_f' + str(self.target_f) + '_mf.tfrecords')) #test
      #fddilename = [os.path.join(self.data_dir, self.subset + '_Dream_50_synth2.tfrecords')] #test
      #filename.append(os.path.join(self.data_dir, self.subset + '_Dream_50_synth2.tfrecords'))
      return filename

    elif self.subset in ['test']:
      #return [os.path.join(self.data_dir, self.subset + '_test_01-02-0001.tfrecords')] #test
      #return [os.path.join(self.data_dir, self.subset + '_test_alpha.tfrecords')] #test
      #return [os.path.join(self.data_dir, self.subset + '_test_01-02-00' + str(self.test_sub[0]) + '_50_100%_f200_mf.tfrecords')] #test
      if self.upsampled:
          return [os.path.join(self.data_dir, 'upsampled_Serial_test_01-02-00' + str(self.test_sub[0]) + '_' + str(self.num_steps) + '_100%_f' + str(self.target_f) + '_from_' + str(self.upsampled) + '_mf.tfrecords')]
      else:
          return [os.path.join(self.data_dir, 'Serial_test_01-02-00' + str(self.test_sub[0]) + '_' + str(self.num_steps) + '_100%_f' + str(self.target_f) + '_mf.tfrecords')] #test
      #return [os.path.join(self.data_dir, self.subset + '_MrOS_sub' + str(self.test_sub[0]) + '_C3_50_100%_f200.tfrecords')]
      #return [os.path.join(self.data_dir, self.subset + '_test_Dino_062014_mPFC.tfrecords')] #test
      #return [os.path.join(self.data_dir, self.subset + '_MASS_sub18_50_100%_f200_mf_power.tfrecords')] #test
      #return [os.path.join(self.data_dir, self.subset + '_excerpt1_detrend.tfrecords')] #test
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses num_step Examples"""
    features = tf.parse_single_example(
    #features = tf.parse_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([self.num_steps*self.num_feats], tf.float32),
        #'label': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.float32),
        })

    #data = tf.decode_raw(features['image'], tf.float32)
    data  = tf.cast(features['image'], tf.float32)
    #label = tf.cast(features['label'], tf.int32)
    label = tf.cast(features['label'], tf.float32)

    #pdb.set_trace()
    data = tf.reshape(data, (self.num_steps,-1))
    mean, std = tf.nn.moments(data,axes=0)
    data1 = []
    #subject_std = [13.8275,14.9942,15.5654,12.9483,12.2577,14.8564,17.8164,19.8062,17.2921,
    #               19.2211,14.5055,19.9593,15.3751,16.2978,14.1444,23.005,15.7555,17.8625,11.5192] #MASS_original data

    # subject_std = [16.5626,14.1226,14.6292,10.9502,15.1776,16.861,15.0652,12.9371] #DREAMS_original data
   

    # 200 Hz:
    subject_std = [13.8275,14.9942,15.5654,12.9483,12.2577,14.8564,17.8164,19.8062,17.2921,
                   19.2211,14.5055,19.9593,15.3751,16.2978,14.1444,23.005,15.7555,17.8625,11.5192]

    # 50 Hz:
    # subject_std = [13.9342, 15.1944,15.946,13.0445,12.4503,15.1196,17.9145,20.0297,17.4839,
                  # 19.606,14.6044,20.0877,15.4103,16.3854,14.1842,23.3035,15.9078,18.0474,11.6242]

     # 34 Hz:
     #subject_std = [13.8587, 15.0691, 15.6965, 12.9845, 12.3263, 14.9442, 17.8637, 19.8735, 17.3554,
     #               19.3543, 14.5438, 20.014, 15.3866, 16.338, 14.167, 23.1198, 15.815, 17.9347, 11.5594]
 
     # 100 Hz:
     # subject_std = [13.9789, 15.26, 16.1699, 13.0276, 12.4902, 15.2348, 17.8223, 20.0905, 17.5904, 19.7739, 14.631,
                     # 20.0389, 15.391, 16.3568, 14.2044, 23.3718, 15.9692, 18.0525, 11.6305]

    #subject_std = [19.2770] #DREAMS_synthetic data
    #pdb.set_trace()
    for k in range(self.num_feats):
      if k==0:
          if self.subset in ['test']:
            data1.append((data[:,k]-mean[k])/np.asarray(subject_std)[int(self.test_sub[0])-1])  # MASS: 15.8103  Dream: 17.0197 MASS-sub1-or: 13.8275 MASS-sub1-and: 13.7274  MASS-sub18-and: 16.8832 MASS-sub18-or: 17.8448 Rats 2.0521e-4
          else:
            data1.append((data[:,k]-mean[k])/np.asarray(subject_std)[np.asarray(self.subject,dtype=int)-1].mean())
      else:
        data1.append(data[:,k])

    data1 = tf.stack(data1,axis=1)
    #data = (data-mean)/std
    #label = tf.cast(tf.greater(tf.reduce_sum(labels), self.num_steps-1), tf.int32)
    #pdb.set_trace()
    return data1, label

  def spindle_producer_datasetAPI(self, batch_size):
    """Output a batch of data"""

    filenames = self.get_filenames()

    dataset = tf.data.TFRecordDataset(filenames).repeat()
    #pdb.set_trace()
    dataset = dataset.map(self.parser, num_parallel_calls=batch_size) #, output_buffer_size=2*batch_size)

    if self.subset == 'train' or self.subset == 'valid':
      min_queue_examples = int(SpindleDatasetBatcher.num_examples_per_epoch(self.subset) * 0.4)
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3*batch_size)

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    data_batch, label_batch = iterator.get_next()
    #pdb.set_trace()
    return data_batch, label_batch

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 5000000 #000
    elif subset == 'valid':
      return 1000000
    elif subset == 'test':
      return 500000
    else:
      raise ValueError('Invalid data subset "%s"' % subset)
