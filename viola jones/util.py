# encoding: utf-8
#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :2/2/19 10:02 PM
#@Author  :Yun Dai
#@FileName: util.py


from mytype import *
from typing import *
from scipy.misc import imread
from glob2 import glob
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np


def merge_boxes(boxes: List[Tuple], thres=3, cluster_thres=1):
    def can_merge(box1: Tuple, box2: Tuple, thres=3):
        return box1[0] - thres <= box2[0] <= box1[2] + thres \
               and box1[0] - thres <= box2[2] <= box1[2] + thres \
               and box1[1] - thres <= box2[1] <= box1[3] + thres \
               and box1[1] - thres <= box2[3] <= box1[3] + thres

    def merge(box1: Tuple, box2: Tuple):
        return min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])

    cluster = {}
    for i, box in enumerate(boxes):
        merge_boxes = []
        for b in cluster:
            if can_merge(b, box, thres):
                merge_boxes.append(b)
        if len(merge_boxes) > 0:
            b = box
            cnt = 0
            for mb in merge_boxes:
                cnt += cluster.pop(mb)
                b = merge(b, mb)
            cluster[b] = cnt
        else:
            cluster[box] = 1
    print('after merge:', len(cluster))
    return dict(filter(lambda x: x[1] >= cluster_thres, cluster.items()))


def load_data(faces_dir, background_dir):
    '''

    :param faces_dir:
    :param background_dir:
    :return: (N x 64 x 64 array, N x 1 array) imgs, label
    '''
    face_files = glob(faces_dir + "/*.jpg")
    bg_files = glob(background_dir + "/*.jpg")
    faces = np.vstack((imread(f)[None,] for f in face_files))
    bgs = np.vstack((imread(f)[None,] for f in bg_files))
    label = np.array([1 for _ in range(len(faces))] + [-1 for _ in range(len(bgs))])
    imgs = np.vstack((faces, bgs)).mean(axis=3)  # to gray scale
    return imgs, label


def draw_face_box(img, bboxs: List[Tuple[int]], line_width=3):
    r, c = img.shape
    img = np.stack((img, img, img), axis=2) / 255.  # to RGB
    m = np.copy(img)
    color = np.array([.9, .0, .0])
    for up_left_r, up_left_c, low_right_r, low_right_c in bboxs:
        m[max(up_left_r - line_width, 0):min(up_left_r + line_width, r), up_left_c:low_right_c, :] = color
        m[max(low_right_r - line_width, 0):min(low_right_r + line_width, r), up_left_c:low_right_c, :] = color
        m[up_left_r:low_right_r, max(up_left_c - line_width, 0):min(up_left_c + line_width, c), :] = color
        m[up_left_r:low_right_r, max(low_right_c - line_width, 0):min(low_right_c + line_width, c), :] = color
    plt.imshow(m, cmap='gray')
    plt.show()
    plt.savefig('./pred_results.png')
