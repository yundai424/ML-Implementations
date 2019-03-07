#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :3/1/19 11:18 AM
#@Author  :Yun Dai
#@FileName: runner.py

import numpy as np
import matplotlib.pyplot as plt
from config import get_config
from MLP import MLP
from CNN import CNN
import sys
from util import *

def get_model(config):
    if config.model_type == 'MLP':
        return MLP(config)
    elif config.model_type == 'CNN':
        return CNN(config)

def run():
    # import config
    config = get_config()
    verbose = config.verbose
    holdout_ratio = config.holdout_ratio
    hid_neurons = config.hid_neurons
    batch_size = config.batch_size

    # load data
    (train_imgs, train_labels), (holdout_imgs, holdout_labels), (test_imgs, test_labels) = \
        load_data(config.train_img_path, config.train_label_path,
                  config.test_img_path, config.test_label_path, config.holdout_ratio, config.maxrow)
    print('Data loaded, batch size = %d' % batch_size)

    # construct model
    model = get_model(config)
    # train model
    print('Begin training...')
    epoch, err = train(train_imgs, train_labels, holdout_imgs, holdout_labels, test_imgs, test_labels, model, config)
    print(epoch, err)

    for filename in config.predict_files:
        print('Begin predicting on %s...' % filename)
        result = predict(model, filename)
        out_file = '.' + filename.split('.')[1] + '.label'
        with open(out_file, 'w') as out:
            for i in result:
                out.write(str(i) + '\n')

def predict(model, file_path):
    imgs = np.loadtxt(file_path, delimiter=',')
    imgs = np.hstack((imgs, np.ones(imgs.shape[0])[:, None]))
    return model.predict_label(imgs)


def train(train_imgs, train_labels, holdout_imgs, holdout_labels, test_imgs, test_labels, model, config):
    epoch = 0
    prev_ho_err = sys.maxsize
    err = 0
    counter = 0
    while epoch < config.max_epoches:
        batch_xs_gen, batch_ys_gen = batch_generator(train_imgs, train_labels, config.batch_size)
        try:
            while True:
                model.minibatch_train(next(batch_xs_gen), next(batch_ys_gen), config.learning_rate, config.experiment)
        except StopIteration:
            pass
        err = model.test_err(test_imgs, test_labels)
        
        if epoch % 10 == 0:
            print('Epoch %d | test err: %.5f' % (epoch, err))
        
        ho_err = model.test_err(holdout_imgs, holdout_labels)
        if ho_err >= prev_ho_err:
            counter += 1
        else:
            counter = 0
        if counter > 2:
            print('Epoch %d | test err: %.5f' % (epoch, err))
            break
        epoch += 1
        prev_ho_err = ho_err
    return epoch, err

def experiment_worker(learning_rates, batch_sizes, layers):
    '''
    experiment on hyperparameters
    :return:
    '''
    config = get_config()
    (train_imgs, train_labels), (holdout_imgs, holdout_labels), (test_imgs, test_labels) = \
        load_data(config.train_img_path, config.train_label_path,
                  config.test_img_path, config.test_label_path, config.holdout_ratio, config.maxrow)

    # learning rates
    res_lr = []
    for learning_rate in learning_rates:
        config_temp = get_config()
        config_temp.learning_rate = learning_rate
        model = get_model(config_temp)
        res_lr.append(train(train_imgs, train_labels, holdout_imgs, holdout_labels, test_imgs, test_labels, model, config_temp))

    # batch_sizes
    res_batch = []
    for batch_size in batch_sizes:
        config_temp = get_config()
        config_temp.batch_size = batch_size
        model = get_model(config_temp)
        res_batch.append(train(train_imgs, train_labels, holdout_imgs, holdout_labels, test_imgs, test_labels, model, config_temp))

# layers
    res_layer = []
    for hid_neurons in layers:
        config_temp = get_config()
        config_temp.hid_neurons = hid_neurons
        model = get_model(config_temp)
        res_layer.append(train(train_imgs, train_labels, holdout_imgs, holdout_labels, test_imgs, test_labels, model, config_temp))

    return res_lr, res_batch, res_layer

def experiment():
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
    batch_sizes = [5, 10, 25, 50, 75, 100]
    layers = [[32], [64], [128], [256], [256, 128], [256, 128, 64]]
    res_lr, res_batch, res_layer = experiment_worker(learning_rates, batch_sizes, layers)
    lr_epochs, lr_errs = [i[0] for i in res_lr], [i[1] for i in res_lr]
    batch_epochs, batch_errs = [i[0] for i in res_batch], [i[1] for i in res_batch]
    layer_epochs, layer_errs = [i[0] for i in res_layer], [i[1] for i in res_layer]
    print(res_lr)
    print(res_batch)
    print(res_layer)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(learning_rates, lr_epochs, color='orange', marker='o', label='epochs')
    ax1.legend(bbox_to_anchor=(0, 0, 0.5, 1))
    ax2.plot(learning_rates, lr_errs, color='darkred', marker='o', label='error rate')
    ax2.legend(bbox_to_anchor=(0, 0, 0.537, 0.9))
    ax1.set_xlabel('learning rate')
    ax1.set_ylabel('epochs')
    ax2.set_ylabel('error rate')
    plt.savefig('./learning_rate.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(batch_sizes, batch_epochs, color='orange', marker='o', label='epochs')
    ax1.legend(bbox_to_anchor=(0, 0, 0.5, 1))
    ax2.plot(batch_sizes, batch_errs, color='darkred', marker='o', label='error rate')
    ax2.legend(bbox_to_anchor=(0, 0, 0.537, 0.9))
    ax1.set_xlabel('learning rate')
    ax1.set_ylabel('epochs')
    ax2.set_ylabel('batch size')
    plt.savefig('./batch.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(layer_epochs, color='orange', marker='o', label='epochs')
    ax1.legend(bbox_to_anchor=(0, 0, 1, 1))
    ax2.plot(layer_errs, color='darkred', marker='o', label='error rate')
    ax2.legend(bbox_to_anchor=(0, 0, 1, 0.9))
    ax1.set_xlabel('learning rate')
    ax1.set_ylabel('epochs')
    ax2.set_ylabel('hidden layers')
    plt.xticks(list(range(6)), ['32', '64', '128', '256', '256,128', '256,128,64'])
    plt.savefig('./layers.png')

if __name__ == '__main__':
    # experiment()
    run()
