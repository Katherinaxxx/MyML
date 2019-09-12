#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/9/5 下午2:31
@Author : Catherinexxx
@Site : 
@File : mnist_gan.py
@Software: PyCharm
"""
from lib.generator.GAN import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('datasets/mnist', one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

#load data
train_X = mnist.train.images                #训练集样本
validation_X = mnist.validation.images      #验证集样本
test_X = mnist.test.images                  #测试集样本
#labels
train_Y = mnist.train.labels                #训练集标签
validation_Y = mnist.validation.labels      #验证集标签
test_Y = mnist.test.labels                  #测试集标签


GAN(data=train_X, x_dimensions=train_X.shape[1], z_dimensions=841, iteration=100000, test_n=20, test_image=False, train=True,
    loss_func='gan', model_path='model/mnist1/gan.ckpt')