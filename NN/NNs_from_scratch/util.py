#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :3/1/19 3:18 PM
#@Author  :Yun Dai
#@FileName: util.py
import numpy as np

def load_data(train_img_path, train_label_path, test_img_path, test_label_path, holdout_ratio, maxrow):
    # load and shuffle data
    imgs = np.loadtxt(train_img_path, delimiter=',', max_rows=maxrow)
    labels = np.loadtxt(train_label_path, delimiter=',', max_rows=maxrow)
    n_total = len(labels)
    permutation = np.random.permutation(n_total)
    imgs = imgs[permutation]
    imgs = np.hstack((imgs, np.ones(imgs.shape[0])[:, None]))
    labels = labels[permutation]

    test_imgs = np.loadtxt(test_img_path, delimiter=',', max_rows=maxrow)
    test_imgs = np.hstack((test_imgs, np.ones(test_imgs.shape[0])[:, None]))
    test_labels = np.loadtxt(test_label_path, delimiter=',', max_rows=maxrow)
    benchmark = int(n_total * holdout_ratio)
    train_imgs = imgs[:benchmark]
    train_labels = labels[:benchmark]
    holdout_imgs = imgs[benchmark:]
    holdout_labels = labels[benchmark:]

    return (train_imgs, train_labels), (holdout_imgs, holdout_labels), (test_imgs, test_labels)

def one_hot(data):
    return (np.arange(10) == data[:, None]).astype(np.integer)

def batch_generator(x, y, batch_size):
    n = len(y)
    permutation = np.random.permutation(n)
    x = x[permutation]
    y = y[permutation]
    y = one_hot(y)
    valid_n = n - n % batch_size
    valid_x = x[:valid_n]
    valid_y = y[:valid_n]
    batch_xs_gen = (i for i in valid_x.reshape(valid_n // batch_size, batch_size, x.shape[-1]))
    batch_ys_gen = (i for i in valid_y.reshape(valid_n // batch_size, batch_size, y.shape[-1]))
    return (batch_xs_gen,
           batch_ys_gen)

def loss(loss_func):
    if loss_func == 'softmax':
        pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# wrapper of activation function
def activation(activation_func):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def relu(x):
        output = np.copy(x)
        output[output < 0] = 0
        return output

    if activation_func == 'sigmoid':
        return sigmoid
    if activation_func == 'relu':
        return relu

def delta_activation(activation_func):
    def delta_sigmoid(x):
        return x * (1.0 - x)
    def delta_relu(x):
        return (x > 0).astype(int)

    if activation_func == 'sigmoid':
        return delta_sigmoid
    if activation_func == 'relu':
        return delta_relu

def softmax_prob(x):
    if x.ndim > 1:
        return np.exp(x) / np.exp(x).sum(axis=1)[:, None]
    else:
        return np.exp(x) / np.exp(x).sum()

def softmax_loss(y, x):
    y_prob = softmax_prob(x)
    return -np.log(y_prob[y.astype(bool)])