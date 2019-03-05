#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :2/24/19 9:18 PM
#@Author  :Yun Dai
#@FileName: tf_cnn.py

import numpy as np
import os
import pathlib
import random
import json
from glob2 import glob
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DataLoader(object):
    @staticmethod
    def _parse_function(filename, label):
      image_string = tf.read_file(filename)
      image_decoded = tf.image.decode_jpeg(image_string, channels=1)
      image_resized = tf.image.resize_images(image_decoded, [64, 64])
      return image_resized, label

    @staticmethod
    def load_data(data_dir, batch_size, ratio, epoch):
        # img file path
        data_root = pathlib.Path(data_dir)
        all_image_paths = list(map(str, data_root.glob('*/*.jpg')))
        random.shuffle(all_image_paths)
        benchmark = len(all_image_paths) * ratio
        train_image_paths = all_image_paths[:benchmark]
        test_image_paths = all_image_paths[benchmark:]
        train_filenames = tf.constant(train_image_paths)
        test_filenames = tf.constant(test_image_paths)
        # assign label
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir() \
                             and 'DS_Store' not in str(item))
        label_to_idx = dict((name, idx) for idx, name in enumerate(label_names))
        all_image_labels = [label_to_idx[pathlib.Path(path).parent.name] for path in all_image_paths]
        train_labels = all_image_labels[:benchmark]
        test_labels = all_image_labels[benchmark:]
        train_labels = tf.constant(train_labels)
        test_labels = tf.constant(test_labels)
        # integration
        train_set = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        test_set = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
        train_set = train_set.map(DataLoader._parse_function)
        test_set = test_set.map(DataLoader._parse_function)
        train_set = train_set.shuffle(buffer_size=1000).batch(batch_size).repeat(epoch)
        print(train_set.output_shapes)
        return train_set, test_set


class CNN(object):
    def __init__(self, config):
        dropout = config['dropout']
        filepath = config['filepath']
        stride1 = config['stride1']
        stride2 = config['stride2']
        convsize1 = config['convsize1']
        convsize2 = config['convsize2']
        learning_rate = config['learning_rate']

        if dropout <= 0 or dropout > 1:
            raise ValueError('Dropout should be float between (0, 1]')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
            self.label = tf.placeholder(tf.int32, [None, 2])
            # construct layers
            conv1 = self.conv2d(self.input,
                                weight = self.init_weight([convsize1, convsize1, 1, 64]),
                                bias = self.init_bias([64]),
                                stride = stride1)
            pool1 = self.maxpool2d(conv1)
            conv2 = self.conv2d(pool1,
                                weight = self.init_weight([convsize2, convsize2, 64, 32]),
                                bias = self.init_bias([32]),
                                stride = stride2)
            pool2 = self.maxpool2d(conv2)
            final_size = int(64 / (stride1 * stride2))
            flatten = tf.reshape(pool2, [-1, final_size * final_size * 32])
            # fully connected 1
            fc1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
            keep_prob = tf.placeholder(tf.float32)
            dropout1 = tf.nn.dropout(fc1, keep_prob) if dropout < 1 else fc1
            # fully connected 2
            fc2 = tf.layers.dense(dropout1, 2)
            y = tf.nn.softmax(fc2)

            # loss func and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc2,
                                                                                   labels=self.label))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            self.correct_pred = tf.equal(tf.argmax(self.label, 1), tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def init_weight(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def init_bias(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, weight, bias, stride=1):
        # Conv2D wrapper with relu activation
        x = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, bias)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2d wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.model = CNN(config)
        self.graph = self.model.graph
        self.train_set, self.test_set = DataLoader.load_data(data_dir=config['filepath'],
                                                             ratio=config['ratio'],
                                                             batch_size=config['batch_size'],
                                                             epoch=config['epoch'])
        self.epoch = config['epoch']

    def run(self):
        with self.graph.as_default(), tf.Session() as sess:
            model_path = os.path.join(self.config.model_path, self.config.model_name)
            saver = tf.train.Saver()
            files = glob(os.path.join(self.config.model_path, '*.ckpt*'))

            if len(files) > 0:
                saver.restore(sess, model_path)
                print(('Model restored from:' + model_path))
            else:
                print("Model doesn't exist.\nInitializing........")
                sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            train_iter = self.train_set.make_initializable_iterator()
            try:
                while True:
                    train_input, train_label = train_iter.get_next()
                    _, cost, accu = sess.run([self.model.optimizer, self.model.cost, self.model.accuracy]


    def test(self):
        with self.graph.as_default(), tf.Session() as sess:
            files = glob(os.path.join(self.config.model_path, '*.ckpt*'))
            assert len(files) > 0
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.config.model_path, self.config.model_name))
            print(('Model restored from:' + self.config.model_path + self.config.model_name))
            self._test(sess)

    def _test(self, sess):
        test_iter = self.test_set.make_one_shot_iterator()
        test_input, test_label = test_iter.get_next()
        accu = sess.run(self.model.accuracy,
                        feed_dict={self.model.input: test_input,
                                   self.model.label: test_label})
        print('test accuracy:%f' % accu)


if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        config = json.load(f)
    runner = Runner(config)
    runner.run()