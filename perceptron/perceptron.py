#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :1/26/19 3:32 PM
#@Author  :Yun Dai
#@FileName: perceptron.py

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    Perceptron classifier
    """
    def __init__(self, dim):
        self.w = np.zeros(dim)
        self.mistake = 0
        self.mistakes = []

    def train(self, digits, label):
        """
        train the classifier (w) with training set
        :param digits: a 2-D array with each line representing a data point
        :param label: a 1-D array with each element representing a label
        :return: None
        """
        n = len(label)
        for t in range(n):
            predict = 1 if np.dot(self.w, digits[t]) >= 0 else -1
            if label[t] == -1 and predict == 1:
                self.w = self.w - digits[t]
                self.mistake += 1
            elif label[t] == 1 and predict == -1:
                self.w = self.w + digits[t]
                self.mistake += 1
            self.mistakes.append(self.mistake)

    def fit(self, digits):
        predicted = np.zeros(len(digits)) - 1
        predicted[np.dot(self.w.T, digits.T) >= 0] = 1
        return predicted

    def error(self, digits, labels):
        return (self.fit(digits) != labels).mean()


def optimize_M(digits, labels, parts=10):
    unit_length = int(len(labels)/parts)
    digits_parts = []
    labels_parts = []
    for i in range(parts):
        digits_parts.append(digits[i * unit_length : (i + 1) * unit_length])
        labels_parts.append(labels[i * unit_length: (i + 1) * unit_length])

    cverrs = np.zeros(10)
    for M in range(1,11):
        cverrs[M - 1] = cross_validation(digits_parts, labels_parts, M)
    return cverrs.argmin() + 1

def cross_validation(digits_parts, labels_parts, M):
    parts = len(digits_parts)
    hoerrs = np.zeros(parts)
    for holdout in range(parts):
        perceptron = Perceptron(len(digits_parts[holdout][0]))
        # feed repeatedly by M times
        for _ in range(M):
            for j in range(parts):
                if j != holdout:
                    perceptron.train(digits_parts[j], labels_parts[j])
        hoerrs[holdout] = perceptron.error(digits_parts[holdout], labels_parts[holdout])
    return hoerrs.mean()

def parse_digits(file):
    digits_list = []
    with open(file, 'r') as digits_file:
        for line in digits_file:
            digits_list.append([1]+list(map(int, line.split())))
    return np.array(digits_list)

def parse_labels(file):
    labels_list = []
    with open(file, 'r') as labels_file:
        for line in labels_file:
            labels_list.append(int(line))
    return np.array(labels_list)

if __name__ == '__main__':
    digits = parse_digits('./train35.digits')
    labels = parse_labels('./train35.labels')
    test_digits = parse_digits('./test35.digits')

    M = optimize_M(digits, labels)
    perceptron = Perceptron(len(digits[0]))
    for _ in range(M):
        perceptron.train(digits, labels)

    test_labels = perceptron.fit(test_digits).astype(int)
    with open('./test35.labels', 'w') as out:
        for i in test_labels:
            out.write(str(i) + '\n')

    plt.plot(perceptron.mistakes)
    plt.xlabel('Number of Examples Seen')
    plt.ylabel('Number of Mistakes')
    plt.title('Cumulative Number of Mistakes')
    plt.savefig('./cumulative_mistakes.png')







