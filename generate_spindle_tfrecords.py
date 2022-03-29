from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import struct
from numpy import loadtxt
import numpy as np
import tensorflow as tf
import pdb
from scipy import signal

SPINDLE_LOCAL_FOLDER = '/data/spindle/SleepSpindleData4RNN/'
SPINDLE_TEST_FOLDER = '/data/spindle/SleepSpindleData4RNN'

def _read_data(filename):
  with open(filename) as f:
    numbers_str = f.read().split()
    numbers_float = [float(x) for x in numbers_str]
  return numbers_float

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def spindle_raw_data(data_path=None, subjects=None, num_steps=50, target_f=200, test_flag=0, upsampled=0):
  file_names = {}
  for subject in subjects:
    file_names[subject] = {}
    if test_flag:

      if upsampled:
        file_names[subject]['data'] = ['upsampled_test_01-02-00' + str(subject) + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_from_' + str(upsampled) + '_mf']
        file_names[subject]['label'] = ['test_01-02-00' + str(subject) + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_mf']
      else:
        file_names[subject]['data'] = file_names[subject]['label'] = ['test_01-02-00' + str(subject) + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_mf']
    else:
      file_names[subject]['train'] = ['Augment_train_MASS_sub' + str(subject) + '_' + str(num_steps) +'_100%_f' + str(target_f) + '_mf']
      file_names[subject]['valid'] = ['Augment_valid_MASS_sub' + str(subject) + '_' + str(num_steps) +'_100%_f' + str(target_f) + '_mf']
      file_names[subject]['test'] = ['Augment_test_MASS_sub' + str(subject) + '_' + str(num_steps) +'_100%_f' + str(target_f) + '_mf']
  return file_names


def convert_to_tfrecord(data_files, label_files, output_file, num_steps, test_flag):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)

  with tf.python_io.TFRecordWriter(output_file) as record_writer:

    for idx in enumerate(data_files):

      print('Working on %s' % data_files[idx[0]])
      print('Working on %s' % label_files[idx[0]])

      #data = _read_data(data_files[idx[0]])
      #label = _read_data(label_files[idx[0]])

      #data = loadtxt(data_files[idx[0]])
      label = loadtxt(label_files[idx[0]])
      feat = [0,1,2,3]
      feat.extend(range(6,25))
      if test_flag:
        with open(data_files[idx[0]]) as infile:
          data = np.zeros([num_steps, 25])
          cnt = 0
          for line in infile:
            line = line.split()
            data[0:num_steps-1, :]=data[1:num_steps, :]
            data[num_steps-1,:]=line
            data1 = data
            data1[:,0] = signal.detrend(data1[:,0], axis=0)
            write_to_tfrecord(data1[:,feat], label[cnt:cnt+num_steps], num_steps, record_writer)
            cnt+=1
      else:
        with open(data_files[idx[0]]) as infile:
          data = []
          cnt = 1
          for line in infile:
            data.append(line.split())
            if cnt%num_steps==0:
              data = np.array(data, dtype=float)
              data.reshape(data.shape[0], -1)
              #data = signal.detrend(data, axis=0)
              write_to_tfrecord(data[:,feat], label[cnt-num_steps:cnt], num_steps, record_writer)
              data = []
            cnt=cnt+1

def write_to_tfrecord(sequence, labels, num_steps, record_writer):

  # If half of the samples in a sequence are a spindle then it gets a "1" label?
  if sum(labels) > num_steps/2:
     Label = 1
  else:
     Label = 0

  example = tf.train.Example(features=tf.train.Features(
    feature={
        'image': _float_feature(sequence.flatten()),
        'label': _float_feature([Label])
    }))

  record_writer.write(example.SerializeToString())

def main(data_path, target_f=200, test_flag=0, upsampled=0):

  subject =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
  test_subject = ['05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']
  
  num_steps = int((250*0.001) / (1 / target_f))

  if test_flag:
    file_names = spindle_raw_data(data_path, test_subject, num_steps, target_f, test_flag, upsampled)
  else:
    file_names = spindle_raw_data(data_path, subject, num_steps, target_f, test_flag)
  

  if test_flag:
    for subject in file_names:
      data_files = [os.path.join(data_path, file_names[subject]['data'][0] + '.txt')]
      label_files = [os.path.join(data_path, file_names[subject]['label'][0] + '_labels.txt')]
      if upsampled:
        output_name = 'upsampled_Serial_test_01-02-00' + subject + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_from_' + str(upsampled) + '_mf.tfrecords'
      else:
        output_name = 'Serial_test_01-02-00' + subject + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_mf.tfrecords'
      output_file = os.path.join(SPINDLE_TEST_FOLDER, output_name)
      convert_to_tfrecord(data_files, label_files, output_file, num_steps, test_flag)
  else:
    for subject in file_names:
      for mode, files in file_names[subject].items():
        data_files = [os.path.join(data_path, f + '.txt') for f in files]
        label_files = [os.path.join(data_path, f + '_labels.txt') for f in files]
        #output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_Dream_nb_50_100%_base.tfrecords')
        # output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_excerpt8_mf.tfrecords')
        #output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_test_01-02-0016_50_100%_f200_mf.tfrecords')
        #output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_test_alpha.tfrecords')
        #output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_test_Dino_062014_mPFC.tfrecords')
        #output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_MASS_sub1_50_100%_f200_less4_1.tfrecords')
        #output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_Dream_50_synth1.tfrecords')
        output_file = os.path.join(SPINDLE_LOCAL_FOLDER, mode + '_MASS_sub' + str(subject) + '_' + str(num_steps) + '_100%_f' + str(target_f) + '_mf.tfrecords')
        # output_file = os.path.join(data_path, SPINDLE_LOCAL_FOLDER + '/' + mode + '_test_synthetic_-5dB.tfrecords')
        convert_to_tfrecord(data_files, label_files, output_file, num_steps, test_flag)

  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_path',
      type=str,
      default='SleepSpindleData4RNN/',
      help='Directory to download and extract CIFAR-10 to.')
  parser.add_argument(
    '--target_f',
    type=int,
    default=50,
    help='Resampled frequency')
  parser.add_argument(
    '--test_flag',
    type=int,
    default=0,
    help='Generate data in test or train mode')
  parser.add_argument(
  '--upsampled',
  type=int,
  default=0,
  help='Frequency from which data was upsampled')

  args = parser.parse_args()
  main(args.data_path, args.target_f, args.test_flag, args.upsampled)

