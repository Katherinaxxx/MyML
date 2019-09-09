#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/9/3 下午4:30
@Author : Catherinexxx
@Site : 
@File : util.py
@Software: PyCharm
"""

import tensorflow as tf
import numpy as np


def next_batch(train_data, batch_size):
    index = [i for i in range(0, len(train_data))]
    np.random.shuffle(index)
    list = index[:batch_size]
    train_data = np.array(train_data)
    batch_data = train_data[list]
    return batch_data


# 构造可训练参数
def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)


# 定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding="SAME", name="atrous_conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义反卷积层
def deconv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="deconv2d"):
    input_dim = input_.get_shape()[-1]
    # input_height = int(input_.shape[1])
    # input_width = int(input_.shape[2])
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim],
                                        [1, 2, 2, 1], padding="SAME")
        return output


# 定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        return output


# 定义lrelu
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)