# encoding: utf-8
#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :2/2/19 11:35 PM
#@Author  :Yun Dai
#@FileName: adaboost.py

'''
@desc:
using adaboost + haar-like filter to play with face recognition data
boosting on decision stumps to find a very small number of relevant features
classifier cascade to drive down false positive rate

Use the Haar-like features described in the Viola–Jones paper and simple decision stumps (with varying
threshold) as weak learners. Each round of boosting will select a single (i,p,θ) combination (where i
indexes the feature, p ∈ {−1, +1} is the polarity, and θ is the threshold) to add to classifier.

'''

from typing import *
import multiprocessing
from operator import add
from functools import reduce
import numpy as np
import pickle
import time
import datetime
import ctypes
import click
import json
from tqdm import tqdm
from mytype import *
from haar import *
from util import *

# for shared memory
global_arr_base = None
shared_integral_array = None


class Adaboost:
    def __init__(self):
        self.model = None
        self.integral = None
        self.label = None

    @classmethod
    def opt_p_theta(self, integral, label, feature_list, weights, feature_idx, enhance = True):
        '''

        :param integral:  N x r x c integral image
        :param label: label Nx1 array of 1 or -1
        :param feature_list:
        :param weights: N, weight of data
        :param feature_idx:
        :param enhance:
        :return: (p, theta)
        '''

        feature = Haar.compute_feature(integral, feature_list, feature_idx)
        sort_idx = np.argsort(feature)
        sorted_label = label[sort_idx]
        sorted_feature = feature[sort_idx]
        signed_sorted_weight = weights[sort_idx] * sorted_label

        # enhanced algorithm that performs exactly the same calculation as the standard way but much faster
        if enhance:
            weight_accu_sum = np.cumsum(signed_sorted_weight)
            min_idx, max_idx = np.argmin(weight_accu_sum), np.argmax(weight_accu_sum)
            theta_p_positive, theta_p_negative = sorted_feature[min_idx] + 1e-3, sorted_feature[max_idx] + 1e-3

            return (1, theta_p_positive) if np.abs(weight_accu_sum[min_idx]) > np.abs(weight_accu_sum[max_idx]) \
                else (-1, theta_p_negative)
        
        # standard calculation as described in the slides
        else:
            pos_idx = np.where(sorted_label > 0)[0]
            neg_idx = np.where(sorted_label < 0)[0]
            t_pos = np.sum(signed_sorted_weight[pos_idx])
            t_neg = np.sum(signed_sorted_weight[neg_idx])
            n = len(sorted_label)
            epsilons = np.zeros(n).astype(float)
            p_s = np.zeros(n)
            for j in range(n):
                s_pos = np.sum(signed_sorted_weight[pos_idx[pos_idx < j]])
                s_neg = np.sum(signed_sorted_weight[neg_idx[neg_idx < j]])
                epsilons[j] = np.min((s_pos + t_neg - s_neg, s_neg + t_pos - s_pos))
                p_s[j] = np.argmin((s_pos + t_neg - s_neg, s_neg + t_pos - s_pos))
            opt_j = np.argmin(epsilons)
            p_s = p_s * 2 - 1
            return (p_s[opt_j], feature[sort_idx[j]] + 1e-3)


    @classmethod
    def _weak_predict(self, integral, feature_list, feature_idx, p, theta):
        '''
        give prediction given a weak learner
        :return: N x 1 array of 1 or -1
        '''
        return np.sign((Haar.compute_feature(integral, feature_list, feature_idx) - theta) * p)

    @classmethod
    def _error_rate(self, integral, label, feature_list, weights, feature_idx, p, theta):
        '''
        compute error rate given a weak learner
        :return: float: err rate
        '''
        prediction = self._weak_predict(integral, feature_list, feature_idx, p, theta)
        diff_idx = np.where(prediction != label)[0]
        return 0 if len(diff_idx) == 0 else weights[diff_idx].sum()

    @classmethod
    def _div_list(self, l, n):
        t = len(l) // n
        parts = [t * i for i in range(0, n)]
        result = [l[parts[i]:parts[i + 1]] for i in range(0, n - 1)]
        result.append(l[parts[n - 1]:])
        return result

    @classmethod
    def _opt_weaklearner_worker(cls, integral, label, weights, feature_list):
        if integral is None:
            global shared_integral_array
            integral = shared_integral_array
        best_p, best_theta, best_feat_idx = None, None, None
        best_err = float('inf')
        for feat_idx in range(len(feature_list)):
            p, theta = cls.opt_p_theta(integral, label, feature_list, weights, feat_idx)
            err = cls._error_rate(integral, label, feature_list, weights, feat_idx, p, theta)
            if err < best_err:
                best_p, best_theta, best_err, best_feat_idx = p, theta, err, feat_idx
        return best_err, best_p, best_theta, best_feat_idx

    @classmethod
    def _opt_weaklearner(cls, integral, label, weights, feature_list, process_num=4):
        ''' compute optimal weak learner (i.e, p, theta,i that minimize err)
        :param integral:
        :param label:
        :param weights:
        :param feature_list:
        :return: (float,float, int) : p,theta, feature_idx
        '''
        if process_num == 1:
            return cls._opt_weaklearner_worker(integral, label, weights, feature_list)
        result = []
        process_num = min(process_num, multiprocessing.cpu_count())
        pool = multiprocessing.Pool(process_num)
        # div list
        each_length = len(feature_list) // process_num
        parts = [each_length * i for i in range(0, process_num)]
        parts.append(len(feature_list))
        t1 = time.time()
        for i in range(process_num):
            result.append(
                pool.apply_async(cls._opt_weaklearner_worker,
                                 (None, label, weights, feature_list[parts[i]:parts[i + 1]])))

        pool.close()
        pool.join()
        print(time.time() - t1)
        best_p, best_theta, best_feat_idx = None, None, None
        best_err = float('inf')
        for i, res in enumerate(result):
            err, p, theta, feat_idx = res.get()
            if err < best_err:
                best_p, best_theta, best_err, best_feat_idx = p, theta, err, feat_idx + i * each_length
        return best_err, best_p, best_theta, best_feat_idx

    def _update_data_weights(self, weights, err, alpha, y_pred, y_true, penalty_factor=1.):
        Z = 2 * np.sqrt((err * (1 - err)))
        data_weight = weights * np.exp(-alpha * y_pred * y_true) / Z
        if penalty_factor > 1:
            diff = y_pred - y_true
            data_weight[diff == 2] *= penalty_factor
            data_weight /= data_weight.sum()
        return data_weight

    @classmethod
    def compute_metrics(self, y_pred, label):
        err_idx, correct_idx = np.where(y_pred != label)[0], np.where(y_pred == label)[0]
        err_label, correct_label = label[err_idx], label[correct_idx]
        fp_cnt, fn_cnt = len(np.where(err_label == -1)[0]), len(np.where(err_label == 1)[0])
        tp_cnt = len(np.where(correct_label == 1)[0])
        fp = fp_cnt / len(np.where(label == -1)[0])  # lower than 30% should be enough
        # we want FN to be as low as possible for each strong classifier in cascade
        fn = fn_cnt / len(np.where(label == 1)[0])
        recall = tp_cnt / len(np.where(label == 1)[0])  # this as high as possible
        accuracy = len(np.where(y_pred == label)[0]) / len(label)
        return Metrics(fn, fp, recall, accuracy)

    def train(self, imgs, label, r_stride, c_stride, min_size, max_size, zoom, niter=20, target_fn=.01, target_fp=.3,
              thres=.3, penalty_on_neg=0., process_num=1):
        process_num = min(multiprocessing.cpu_count(), process_num)
        print('%d cpu used.' % process_num)
        print('adaboost training start!')
        n, r, c = imgs.shape
        print('computing integral imgs....')
        integral = Haar.haar_integral_image(imgs)
        print('computing features....')
        feature_list = Haar.get_2rec_featureH(row=64, col=64,
                                              r_stride=r_stride, c_stride=c_stride, zoom=zoom,
                                              min_size=min_size,
                                              max_size=max_size)
        feature_list += Haar.get_2rec_featureV(row=64, col=64,
                                               r_stride=r_stride, c_stride=c_stride, zoom=zoom,
                                               min_size=min_size,
                                               max_size=max_size)
        feature_list += Haar.get_3rec_feature(row=64, col=64,
                                              r_stride=r_stride, c_stride=c_stride, zoom=zoom,
                                              min_size=min_size,
                                              max_size=max_size)
        feature_list += Haar.get_4rec_feature(row=64, col=64,
                                              r_stride=r_stride, c_stride=c_stride, zoom=zoom,
                                              min_size=min_size,
                                              max_size=max_size)
        print("%d features in total..." % len(feature_list))
        data_weights = np.ones((n,), np.float64) / len(imgs)
        # since we only have 2000/2000 pos and neg samples
        # we might want to add some penalty on neg sample since we have less and less neg samples in deeper cascade
        penalty = 1.
        if penalty_on_neg > 0:
            pos_cnt, neg_cnt = len(np.where(label == 1)[0]), len(np.where(label == -1)[0])
            penalty = max(float(pos_cnt / neg_cnt) - 1, 0) * penalty_on_neg + 1
            print('pos %d, neg %d, penalty on neg %f' % (pos_cnt, neg_cnt, penalty))
            data_weights[label == -1] *= penalty
            data_weights /= data_weights.sum()

        model = Model([], feature_list, r, c, thres)
        # for multiprocess shared memory
        if process_num > 1:
            global global_arr_base
            global shared_integral_array
            n, r, c = integral.shape
            global_arr_base = multiprocessing.Array(ctypes.c_double, integral.ravel())
            shared_integral_array = np.ctypeslib.as_array(global_arr_base.get_obj())
            shared_integral_array = shared_integral_array.reshape(n, r, c)
        for i in range(niter):
            print('\niter %d --------------------------------' % i)
            err, p, theta, feat_idx = self._opt_weaklearner(integral, label, data_weights, feature_list, process_num)
            y_weak_pred = self._weak_predict(integral, feature_list, feat_idx, p, theta)
            alpha = 1 if err == 0 else .5 * np.log((1 - err) / err)  # update learner weight
            model.weak_learners.append(Learner(feat_idx, p, theta, alpha))
            y_pred = self.predict(model, None, thres, integral)
            data_weights = self._update_data_weights(data_weights, err, alpha, y_weak_pred, label,
                                                     penalty if penalty_on_neg > 0 else 1)
            metrics = self.compute_metrics(y_pred, label)
            # print('weak learner', model.weak_learners[-1])
            # print("learner err rate %f" % (err,))
            print('metrics', metrics)
            if err == 0 or (metrics.fn < target_fn and metrics.fp < target_fp):
                break
        print('stop training.')
        model.frozen()
        self.model = model
        return model

    @classmethod
    def predict(cls, model: Model, imgs, thres=float('nan'), integral=None):
        '''
        :return: N x 1
        '''
        if np.isnan(thres):
            thres = model.thres
        if integral is None:
            integral = Haar.haar_integral_image(imgs)
        weak_learners, feature_list = model.weak_learners, model.feature_list
        normalizer = sum(learner.weight for learner in weak_learners)
        res = reduce(add, map(
            lambda learner: learner.weight * cls._weak_predict(integral, feature_list, learner.i, learner.p,
                                                               learner.theta),
            weak_learners)) / normalizer
        # larger thres so that we won't miss any potential face in cascade
        # print(res + thres)
        return np.sign(res + thres)

    def save_model(self, path='./model.ada'):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print('model saved to ' + path)

    def load_model(self, path='./model.ada'):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def test(self):
        res = self.predict(self.model, None, 0, self.integral)
        diff = self.label - res
        print(np.count_nonzero(diff) / len(res))
        with open('./test', 'w') as f:
            for i in diff:
                f.write(str(i) + "\n")

    def load_data(self, imgs, label):
        integral = Haar.haar_integral_image(imgs)
        self.integral, self.label = integral, label


class CascadeClassifier:
    def __init__(self):
        self.cascade = []

    def train(self, imgs, label, r_stride, c_stride, min_size, max_size, zoom, niter_each=20, max_cascade=10,
              stop_fp=.02, single_fn=.01, single_fp=.3, thres=.3,
              adaptive_thres=False, penalty_on_neg=0., process_num=4):
        '''

        :param imgs:
        :param label:
        :param niter_each:
        :param max_cascade:
        :param single_fn:
        :param single_fp:
        :param thres:
        :param adaptive_thres: if we want to lower the thres when neg sample become less
        :param penalty_on_neg: if we want to impose penalty on neg sample when it become less and less
        :return:
        '''
        print('classifier training start!')
        n, r, c = imgs.shape
        pos_imgs, neg_imgs = imgs[label == 1], imgs[label == -1]
        pos_label, neg_label = label[label == 1], label[label == -1]
        neg_integral = Haar.haar_integral_image(neg_imgs)
        init_neg_cnt = len(neg_label)
        for i in range(max_cascade):
            print('\nlayer %d\n===========================\n' % i)
            imgs_cur_stage, label_cur_stage = np.vstack((pos_imgs, neg_imgs)), np.hstack((pos_label, neg_label))
            ada = Adaboost()
            cur_thres = thres * float(len(neg_label)) / init_neg_cnt if adaptive_thres else thres
            model = ada.train(imgs=imgs_cur_stage, label=label_cur_stage,
                              r_stride=r_stride, c_stride=c_stride,
                              min_size=min_size, max_size=max_size, zoom=zoom,
                              niter=niter_each, target_fn=single_fn, target_fp=single_fp,
                              thres=cur_thres, penalty_on_neg=penalty_on_neg,
                              process_num=process_num)
            # set fp as N sample in next run
            y_pred = Adaboost.predict(model, None, thres, neg_integral)
            diff = y_pred - neg_label
            fp_idx = np.where(diff == 2)[0]
            print('%d fp in this stage' % len(fp_idx))
            neg_imgs, neg_label, neg_integral = neg_imgs[fp_idx], neg_label[fp_idx], neg_integral[fp_idx]
            self.cascade.append(model)
            if len(fp_idx) / init_neg_cnt < stop_fp:
                print('reach target fp, stop')
                break

        print("classifier training stop")

    def _predict(self, integral, thres=float('nan')):
        '''

        :param integral: N x r x c
        :param thres:
        :return: N x 1
        '''
        apply_thres_delta = np.isnan(thres)
        idx_cur_stage = np.arange(0, len(integral))
        cnt = 0
        for model in self.cascade:
            cnt += 1
            y_pred = Adaboost.predict(model, None, model.thres + thres if apply_thres_delta else thres,
                                      integral=integral[idx_cur_stage])
            pos_idx = np.where(y_pred == 1)[0]
            idx_cur_stage = idx_cur_stage[pos_idx]
            if len(idx_cur_stage) == 0:
                break
        pred = np.full((len(integral)), -1)
        pred[idx_cur_stage] = 1
        # print('%d stage' % cnt)
        return pred

    def __repr__(self):
        return '\n'.join(str(model) for model in self.cascade)

    def save(self, cfg):
        dt = datetime.datetime.today().strftime('%Y%m%d_%H:%M')
        with open('./%s.cascade' % dt, 'wb') as f:
            pickle.dump(self.cascade, f)
        with open('./%s.json' % dt, 'w') as f:
            json.dump(cfg, f)
        print('model saved in ' + './%s.cascade' % dt)

    def load(self, path='./model.cascade'):
        with open(path, 'rb') as f:
            self.cascade = pickle.load(f)

    def _pred_whole_worker(self, integral, row_range, r_stride=2, c_stride=2, thres_delta=float('nan'),
                           verbose=False):
        if integral is None:
            global shared_integral_array
            integral = shared_integral_array
        detected_boxes = []
        _, row, col = integral.shape
        window_row, window_col = self.cascade[0].row_size, self.cascade[0].col_size
        for r in row_range:
            for c in range(0, col - window_col, c_stride):
                if self._predict(integral[:, r:r + window_row, c:c + window_col], thres_delta) > 0:
                    detected_boxes.append((r, c, r + window_row, c + window_col))
                    if verbose:
                        print(r, c, 'detected')
        return detected_boxes

    def pred_whole_img(self, img, r_stride=2, c_stride=2, thres_delta=float('nan'), process_num=1, verbose=False):
        if not self.cascade:
            raise ValueError('model non exist!')
        process_num = min(multiprocessing.cpu_count(), process_num)
        print('%d cpu used' % process_num)
        row, col = img.shape
        window_row, window_col = self.cascade[0].row_size, self.cascade[0].col_size
        integral = Haar.haar_integral_image(img[None,])
        global global_arr_base
        global shared_integral_array
        n, r, c = integral.shape
        global_arr_base = multiprocessing.Array(ctypes.c_double, integral.ravel())
        shared_integral_array = np.ctypeslib.as_array(global_arr_base.get_obj()).reshape(n, r, c)
        row_range = list(range(0, row - window_row, r_stride))
        if process_num > 1:
            each_length = len(row_range) // process_num
            parts = [each_length * i for i in range(0, process_num)]
            parts.append(len(row_range))
            pool = multiprocessing.Pool()
            result = []
            for i in range(process_num):
                result.append(
                    pool.apply_async(self._pred_whole_worker, (
                        None, row_range[parts[i]:parts[i + 1]], r_stride, c_stride, thres_delta,
                        verbose)))

            pool.close()
            pool.join()
            detected_boxes = reduce(add, map(lambda x: x.get(), result))
        else:
            detected_boxes = self._pred_whole_worker(integral, row_range,
                                                     r_stride, c_stride, thres_delta, verbose)
        return detected_boxes


@click.command()
@click.option("-t", "--train", default=False, is_flag=True, help="training mode.")
@click.option("-p", "--predict", default=False, is_flag=True, help="prediction mode.")
@click.option("-v", "--verbose", default=False, is_flag=True)
def run(train, predict, verbose):
    if train:
        print("training mode, loading config from ./train.json")
        with open('./train.json', 'r') as f:
            config = json.load(f)
        print(config)
        imgs, label = load_data(config['face_path'], config['background_path'])
        cascade = CascadeClassifier()
        cascade.train(imgs, label,
                      niter_each=config['niter'],
                      r_stride=config['r_stride'], c_stride=config['c_stride'],
                      min_size=config['min_size'], max_size=config['max_size'],
                      zoom=config['zoom'],
                      max_cascade=config['max_cascade'],
                      stop_fp=config['stop_fp'],
                      single_fn=config['single_fn'],
                      single_fp=config['single_fp'],
                      thres=config['thres'],
                      adaptive_thres=config['adaptive_thres'],
                      penalty_on_neg=config['penalty_on_neg'],
                      process_num=config['process_num']
                      )
        cascade.save(config)
    elif predict:
        print("prediction mode, loading config from ./predict.json")
        with open('./predict.json', 'r') as f:
            config = json.load(f)
        print(config)
        cascade = CascadeClassifier()
        cascade.load(config['model_path'])
        im = imread(config['test_file'], mode='RGB').mean(axis=2)
        # im = imread(config['test_file']).mean(axis=2)
        # print((im-im1).sum())
        detected_boxes = cascade.pred_whole_img(img=im, r_stride=config['r_stride'], c_stride=config['c_stride'],
                                                thres_delta=float(config['thres_delta']),
                                                process_num=config['process_num'], verbose=verbose)

        if config['merge']:
            detected_boxes = merge_boxes(detected_boxes, config['merge_thres'])
        print(len(detected_boxes))
        draw_face_box(im, detected_boxes, config['line_width'])
    else:
        print('pls specify training or prediction mode')
        exit(-1)


if __name__ == '__main__':
    # imgs, label = load_data('./data/face/faces', './data/face/background')
    # np.save('./imgs.npy', imgs)
    # np.save('./label.npy', label)
    # imgs = np.load('./imgs.npy')
    # label = np.load('./label.npy')

    # ada = Adaboost()
    #
    # model = ada.train(imgs=imgs, label=label, niter=10,
    #                   target_fn=.05, target_fp=.05,
    #                   thres=0., penalty_on_neg=0.,
    #                   process_num=4)
    # ada.save_model('./model.ada')
    # for l in ada.model.weak_learners:
    #     print(l)
    run()
