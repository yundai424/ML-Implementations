# encoding: utf-8
#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :2/2/19 11:55 PM
#@Author  :Yun Dai
#@FileName: mytype.py


class Learner:
    __slots__ = ['i', 'p', 'theta', 'weight']

    def __init__(self, i, p, theta, weight):
        self.i, self.p, self.theta, self.weight = i, p, theta, weight

    def __repr__(self):
        return str({k: getattr(self, k) for k in self.__slots__})


class Metrics:
    __slots__ = ['fn', 'fp', 'recall', 'accuracy']

    def __init__(self, fn, fp, recall, accuracy):
        self.fn, self.fp, self.recall, self.accuracy = fn, fp, recall, accuracy

    def __repr__(self):
        return str({k: getattr(self, k) for k in self.__slots__})


class Point:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return str({k: getattr(self, k) for k in self.__slots__})


class HaarFeature:
    __slots__ = ['pos', 'neg']

    def __init__(self, pos, neg):
        self.pos, self.neg = pos, neg

    def __repr__(self):
        return str({k: getattr(self, k) for k in self.__slots__})


class Model:
    __slots__ = ['weak_learners', 'feature_list', 'thres', 'row_size', 'col_size']

    def __init__(self, weak_learners, featurelist, row_size, col_size, thres=0.):
        self.weak_learners, self.feature_list, self.thres = weak_learners, featurelist, thres
        self.row_size, self.col_size = row_size, col_size

    def frozen(self):
        compressed_feature_list = []
        feature_map = {}
        for l in self.weak_learners:
            if not l.i in feature_map:
                feature_map[l.i] = len(compressed_feature_list)
                compressed_feature_list.append(self.feature_list[l.i])
            l.i = feature_map[l.i]
        self.feature_list = compressed_feature_list

    def __repr__(self):
        return str({k: getattr(self, k) for k in self.__slots__})
