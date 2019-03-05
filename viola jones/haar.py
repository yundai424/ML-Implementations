# encoding: utf-8
#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :2/2/19 11:10 PM
#@Author  :Yun Dai
#@FileName: haar.py


from mytype import *
import numpy as np
from typing import *


class Haar:
    @staticmethod
    def haar_integral_image(imgs):
        '''

        :param imgs: N x r x c images
        :return: N x r x c integral images
        '''
        n, row, col = imgs.shape
        integral = np.zeros_like(imgs)
        row_sum = np.zeros((n,))
        for r in range(row):
            for c in range(col):
                row_sum += imgs[:, r, c]
                integral[:, r, c] = row_sum + (0 if r == 0 else integral[:, r - 1, c])
            row_sum.fill(0)
        return integral

    @staticmethod
    def get_2rec_featureV(row=64, col=64, r_stride=2, c_stride=2, zoom=2, min_size=4, max_size=64):
        '''
        type 1 feature
        A  B
        C  D
        E  F
        area = A + 2D + E - B - 2C - F
        :return:
        '''
        features = []
        for w in range(min_size, min(max_size + 1, col), zoom):
            for h in range(min_size * 2, min(2 * (max_size + 1), row), zoom * 2):
                for r in range(0, 1 + row - h, r_stride):
                    for c in range(0, 1 + col - w, c_stride):
                        A, B = Point(r - 1, c - 1), Point(r - 1, c + w - 1)
                        C, D = Point(r - 1 + h // 2, c - 1), Point(r - 1 + h // 2, c + w - 1)
                        E, F = Point(r - 1 + h, c - 1), Point(r - 1 + h, c + w - 1)
                        features.append(HaarFeature(pos=[A, E, D, D], neg=[B, F, C, C]))
        return features

    @staticmethod
    def get_2rec_featureH(row=64, col=64, r_stride=2, c_stride=2, zoom=2, min_size=4, max_size=64):
        '''
        type 2 feature
        A  B  C
        D  E  F
        area = A + C + 2E - 2B - D - F
        :return:
        '''
        features = []
        for w in range(min_size * 2, min(2 * (max_size + 1), col), zoom * 2):
            for h in range(min_size, min(max_size + 1, row), zoom):
                for r in range(0, 1 + row - h, r_stride):
                    for c in range(0, 1 + col - w, c_stride):
                        A, B, C = Point(r - 1, c - 1), Point(r - 1, c + w // 2 - 1), Point(r - 1, c + w - 1)
                        D, E, F = Point(r - 1 + h, c - 1), Point(r - 1 + h, c + w // 2 - 1), Point(r - 1 + h, c + w - 1)
                        features.append(HaarFeature(pos=[A, C, E, E], neg=[B, B, D, F]))
        return features

    @staticmethod
    def get_4rec_feature(row=64, col=64, r_stride=2, c_stride=2, zoom=2, min_size=4, max_size=64):
        '''
        type 3 feature
        A  B  C
        D  E  F
        G  H  I
        area = A + C + 4E + G + I - 2B - 2D - 2F - 2H
        :return:
        '''
        features = []
        for w in range(min_size * 2, min(2 * (max_size + 1), col), zoom * 2):
            for h in range(min_size * 2, min(2 * (max_size + 1), row), zoom * 2):
                for r in range(0, 1 + row - h, r_stride):
                    for c in range(0, 1 + col - w, c_stride):
                        A, B, C = Point(r - 1, c - 1), Point(r - 1, c + w // 2 - 1), Point(r - 1, c + w - 1)
                        D, E, F = Point(r - 1 + h // 2, c - 1), Point(r - 1 + h // 2, c + w // 2 - 1), Point(
                            r - 1 + h // 2, c + w - 1)
                        G, H, I = Point(r - 1 + h, c - 1), Point(r - 1 + h, c + w // 2 - 1), Point(r - 1 + h, c + w - 1)
                        features.append(HaarFeature(pos=[A, C, G, I, E, E, E, E], neg=[B, B, D, D, F, F, H, H]))
        return features

    @staticmethod
    def get_3rec_feature(row=64, col=64, r_stride=1, c_stride=1, zoom=1, min_size=1, max_size=64):
        '''
        type 2 feature
        A  B  C  D
        E  F  G  H
        area = 2B + 2G + D + E - 2C - 2F - A - H
        :return:
        '''
        features = []
        for w in range(min_size * 3, min(3 * (max_size + 1), col), zoom * 3):
            for h in range(min_size, min(max_size, row), zoom):
                for r in range(0, 1 + row - h, r_stride):
                    for c in range(0, 1 + col - w, c_stride):
                        A, B, C, D = Point(r - 1, c - 1), Point(r - 1, c + w // 3 - 1), \
                                     Point(r - 1, c + 2 * w // 3 - 1), Point(r - 1, c + w - 1)
                        E, F, G, H = Point(r - 1 + h, c - 1), Point(r - 1 + h, c + w // 3 - 1), \
                                     Point(r - 1 + h, c + 2 * w // 3 - 1), Point(r - 1 + h, c + w - 1)
                        features.append(HaarFeature(pos=[B, B, G, G, D, E], neg=[C, C, F, F, A, H]))
        return features

    @staticmethod
    def compute_feature(integral: np.ndarray, feature_list: List[HaarFeature], feature_idx):
        '''

        :param integral: N x r x c integral image
        :param feature_list: List(HaarFeature)
        :param feature_idx: int
        :return: N x 1 numpy matrix of the feature evaluations for each image
        '''

        feat = feature_list[feature_idx]
        return sum(integral[:, p.x, p.y] for p in feat.pos if p.x >= 0 and p.y >= 0) - \
               sum(integral[:, p.x, p.y] for p in feat.neg if p.x >= 0 and p.y >= 0)
