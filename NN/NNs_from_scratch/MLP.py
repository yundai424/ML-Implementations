#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :3/1/19 11:18 AM
#@Author  :Yun Dai
#@FileName: MLP.py

import numpy as np
from util import *
from config import get_config

class MLP(object):
    def __init__(self, config):
        self.n_inputs = config.input_dim + 1 # 1 for the bias neuron
        self.hid_neurons = config.hid_neurons
        self.n_outputs = config.n_output
        self.weights = self.init_weight()
        self.activation = activation(config.activation_func)
        self.delta_activation = delta_activation(config.activation_func)

    def init_weight(self):
        '''
        weight matrix from l-1 to l: n_{l-1} * n_{l}. row vector -> O(t), col vector -> I(t)
        '''
        def truncated_norm(n_in, n_out):
            return np.clip(np.random.standard_normal((n_in, n_out)), -2, 2)

        weights = []
        weights.append(truncated_norm(self.n_inputs, self.hid_neurons[0]))
        for l in range(1, len(self.hid_neurons)):
            weights.append(truncated_norm(self.hid_neurons[l - 1], self.hid_neurons[l]))
        weights.append(truncated_norm(self.hid_neurons[-1], self.n_outputs))
        return weights

    def minibatch_train(self, x, y, learning_rate, experiment):
        '''
        :param x: batched training images, dimension: (number of batches, batch size, image vector length)
        :param y: batched one-hot encoded training labels, dimension: (number of batches, batch size, 10)
        :param experiment: boolean, True for parameter tuning
        :return:
        '''
        # x: (batch size, 785)
        deltas, activations = self.back_propagate(x, y)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * activations[i].T @ deltas[i]

    def predict_label(self, x):
        _, y_pred = self.forward(x)
        return y_pred.argmax(axis=1)

    def test_err(self, x, y):
        y_pred = self.predict_label(x)
        return (y_pred != y).mean()

    def forward(self, x):
        '''
        :return: a list of activation vectors, a list of Z vectors and probability output
        '''
        activations = [x]
        for i in range(len(self.weights)):
            x = self.activation(x @ self.weights[i])
            activations.append(x)
        return activations, softmax_prob(x)

    def back_propagate(self, x, y):
        activations, y_pred = self.forward(x)
        deltas = [np.zeros((x.shape[0], weight.shape[-1])) for weight in self.weights]
        deltas[-1] += self.delta_activation(activations[-1]) * (y_pred - y)
        for i in range(len(self.weights) - 2, -1, -1):
            deltas[i] += (self.delta_activation(activations[i + 1]) *
                                     (self.weights[i + 1] @ deltas[i + 1].T).T)
        return deltas, activations