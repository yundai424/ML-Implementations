#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :3/1/19 11:33 AM
#@Author  :Yun Dai
#@FileName: config.py

class Config(object):
    def __init__(self):
        self.model_type = 'MLP'
        self.train_img_path = "./TrainDigitX.csv"
        self.train_label_path = "./TrainDigitY.csv"
        self.test_img_path = "./TestDigitX.csv"
        self.test_label_path = "./TestDigitY.csv"
        self.holdout_ratio = 0.8
        self.verbose = True
        self.learning_rate = 0.1
        self.input_dim = 784
        self.n_output = 10
        self.batch_size = 50
        self.hid_neurons = [256, 128, 64]
        self.activation_func = 'sigmoid'
        self.max_epoches = 100
        self.experiment = False
        self.maxrow = None
        self.predict_files = ["./TestDigitX.csv", "./TestDigitX2.csv"]

def get_config():
    return Config()


