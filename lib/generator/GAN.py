#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/9/3 下午4:27
@Author : Catherinexxx
@Site : 
@File : GAN.py
@Software: PyCharm
"""

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from lib.generator.util import *
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


class GAN(object):

    def __init__(self, data, x_dimensions, z_dimensions, y_dimensions=0, batch_size=32, loss_func='gan',
                 optimizer='adam', d_lr=0.0003, g_lr=0.0001, iteration=1000, test_n=20, train=True, test=True,
                 model_path='model/gan.ckpt', test_path='test.csv', test_image=False):
        """

        :param data: train dataset x
        :param x_dimensions: dimension of x
        :param z_dimensions: dimension of z
        :param y_dimensions: dimension of y
        :param batch_size: batch size
        :param loss_func: 'wgangp_loss', 'lsgan', 'gan', 'hinge'
        :param optimizer: 'adam', 'sgd
        :param d_lr: learning rate of discriminator
        :param g_lr: learning rate of generator
        :param iteration: number of iterations
        :param test_n: number of tests
        :param train: whether to train or not
        :param test: whether to test or not
        :param model_path: model save path
        :param test_path: test save path
        :param test_image: whether to show test image
        """
        self.data = data
        self.x_dimensions = x_dimensions
        self.z_dimensions = z_dimensions
        self.y_dimensions = y_dimensions
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.iteration = iteration
        self.test_n = test_n
        self.train = train
        self.test = test
        self.model_path = model_path
        self.test_path = test_path
        self.test_image = test_image
        self.main()

    def generate_z(self, n, _type='minority'):  # batch；默认生成少数
        z = np.random.rand(n, self.z_dimensions)
        if _type == 'None':
            return z
        if _type == 'minority':
            y = np.zeros([n, self.y_dimensions])
            z_batch = np.hstack((z, y))
            return z_batch
        if _type == 'majority':
            y = np.ones([n, self.y_dimensions])
            z_batch = np.hstack((z, y))
            return z_batch

    def loss(self, logits_real, logits_fake, x, G_sample):
        d_loss, g_loss = 0, 0
        if self.loss_func == 'wgangp':
            """Compute the WGAN-GP loss.

            Inputs:
            - logits_real: Tensor, shape [batch_size, 1], output of discriminator
                Log probability that the image is real for each real image
            - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
                Log probability that the image is real for each fake image
            - batch_size: The number of examples in this batch
            - x: the input (real) images for this batch
            - G_sample: the generated (fake) images for this batch

            Returns:
            - D_loss: discriminator loss scalar
            - G_loss: generator loss scalar
            """

            # compute D_loss and G_loss
            d_loss = - tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
            g_loss = - tf.reduce_mean(logits_fake)

            # lambda from the paper
            lam = 10
            # random sample of batch_size (tf.random_uniform)
            eps = tf.random_uniform([self.batch_size, 1], minval=0.0, maxval=1.0)
            # print(G_sample)
            x_penalty = eps * x + (1 - eps) * tf.reshape(G_sample, [self.batch_size, -1])  # Ppenalty
            # diff = G_sample - x
            # interp = x + (eps * diff)

            # Gradients of Gradients is kind of tricky!
            with tf.variable_scope('', reuse=True) as scope:
                grad_D_x_hat = tf.gradients(self.discriminator(x_penalty, reuse=True), x_penalty)

            grad_norm = tf.norm(grad_D_x_hat[0], axis=1, ord='euclidean')
            grad_pen = tf.reduce_mean(tf.square(grad_norm - 1))
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=[1]))
            # grad_pen = tf.reduce_mean((slopes - 1.) ** 2)

            d_loss += lam * grad_pen

        # if loss_func.__contains__('wgan'):
        #     real_loss = -tf.reduce_mean(real)
        #     fake_loss = tf.reduce_mean(fake)

        if self.loss_func == 'lsgan':
            real_loss = tf.reduce_mean(tf.squared_difference(logits_real, 1.0))
            fake_loss = tf.reduce_mean(tf.square(logits_fake))
            d_loss = real_loss + fake_loss
            g_loss = tf.reduce_mean(tf.squared_difference(logits_fake, 1.0))

        if self.loss_func == 'gan' or self.loss_func == 'dragan':
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
            d_loss = real_loss + fake_loss
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))

        if self.loss_func == 'hinge':
            real_loss = tf.reduce_mean(relu(1.0 - logits_real))
            fake_loss = tf.reduce_mean(relu(1.0 + logits_fake))
            d_loss = real_loss + fake_loss
            g_loss = -tf.reduce_mean(logits_fake)

        return d_loss, g_loss

    def generator(self, image, gf_dim=1, dim=16, reuse=False, name="generator"):
        """
        Four CNN(BN+lrelu) + Four DCNN + one FC
        :param image: [n, dim]
        :param gf_dim: filter
        :param dim: image.get_shape()[1]
        :param reuse:
        :param name:
        :return:[1, p]
        """
        # input_dim = int(image.get_shape()[-1])  # 获取输入通道
        dropout_rate = 0.3  # 定义dropout的比例
        batch_size = image.get_shape()[0]
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            # image = tf.reshape(image, [1, batch_size, -1, 1])

            e1 = batch_norm(tf.layers.max_pooling2d(conv2d(input_=image, output_dim=gf_dim, kernel_size=4, stride=1, name='g_e1_conv'),
                                                    pool_size=[2, 2], strides=2, name='g_conv_p1'), name='g_bn_e1')
            e1 = tf.nn.dropout(e1, dropout_rate)

            e2 = batch_norm(tf.layers.max_pooling2d(conv2d(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=1, name='g_e2_conv'),
                                                    pool_size=[2, 2], strides=2, name='g_conv_p2'), name='g_bn_e2')
            e2 = tf.nn.dropout(e2, dropout_rate)

            e3 = batch_norm(tf.layers.max_pooling2d(conv2d(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=1, name='g_e3_conv'),
                                                    pool_size=[2, 2], strides=2, name='g_conv_p3'), name='g_bn_e3')
            e3 = tf.nn.dropout(e3, dropout_rate)

            e4 = batch_norm(tf.layers.max_pooling2d(conv2d(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=1, name='g_e4_conv'),
                                                    pool_size=[2, 2], strides=2, name='g_conv_p4'), name='g_bn_e4')
            e4 = tf.nn.dropout(e4, dropout_rate)

            d1 = tcl.conv2d_transpose(e4, gf_dim * 8, 4, stride=1,
                                      activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME',
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))
            d1 = tf.nn.dropout(d1, dropout_rate)  # 随机扔掉一般的输出

            d2 = tcl.conv2d_transpose(d1, gf_dim * 4, 4, stride=1,
                                      activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME',
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))

            d2 = tf.nn.dropout(d2, dropout_rate)  # 随机扔掉一般的输出
            d3 = tcl.conv2d_transpose(d2, gf_dim * 2, 4, stride=1,
                                      activation_fn=tf.nn.relu, normalizer_fn=None, padding='SAME',
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))

            d3 = tf.nn.dropout(d3, dropout_rate)  # 随机扔掉一般的输出
            d4 = tcl.conv2d_transpose(d3, 1, 4, stride=1,
                                      activation_fn=None, normalizer_fn=None, padding='SAME',
                                      weights_initializer=tf.random_normal_initializer(0, 0.02))
            d4 = tf.reshape(d4, [batch_size, -1])
            g = tcl.fully_connected(d4, dim, None)
            return g

    def discriminator(self, image, df_dim=2, reuse=False, name="discriminator"):
        """
        Four CNN
        :param image: [n, p]
        :param df_dim:
        :param reuse:
        :param name:
        :return: logit [n, 1]
        """
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            dim = image.get_shape()[0]
            # image = tf.reshape(image, [1, dim, -1, 1])
            h0 = lrelu(conv2d(input_=image, output_dim=df_dim, kernel_size=4, stride=2, name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(input_=h0, output_dim=df_dim * 2, kernel_size=4, stride=2, name='d_h1_conv'),
                                  name='d_bn1'))
            h2 = lrelu(batch_norm(conv2d(input_=h1, output_dim=df_dim * 4, kernel_size=4, stride=2, name='d_h2_conv'),
                                  name='d_bn2'))
            h3 = lrelu(batch_norm(conv2d(input_=h2, output_dim=df_dim * 8, kernel_size=4, stride=1, name='d_h3_conv'),
                                  name='d_bn3'))
            output = conv2d(input_=h3, output_dim=1, kernel_size=4, stride=1, name='d_h4_conv')
            dis_out = tf.sigmoid(output)  # 在输出之前经过sigmoid层，因为需要进行log运算
            return dis_out

    def optimization(self, lr):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

    def main(self):
        z_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.z_dimensions], name='z_placeholder')
        x_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, np.sqrt(self.x_dimensions), np.sqrt(self.x_dimensions), 1], name='x_placeholder')

        g = self.generator(tf.reshape(z_placeholder, [self.batch_size, int(np.sqrt(self.z_dimensions)), int(np.sqrt(self.z_dimensions)), 1]), dim=self.x_dimensions, reuse=False, name="generator")
        Gz = tf.reshape(g, [self.batch_size, int(np.sqrt(self.x_dimensions)), int(np.sqrt(self.x_dimensions)), 1])
        Dx = self.discriminator(x_placeholder, reuse=False, name="discriminator")
        Dg = self.discriminator(Gz, reuse=True, name="discriminator")

        d_loss, g_loss = self.loss(Dx, Dg, x_placeholder, g)

        g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]  # 所有生成器的可训练参数
        d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]  # 所有判别器的可训练参数

        global_ = tf.placeholder(dtype=tf.int32)
        d_train = self.optimization(self.d_lr).minimize(d_loss, var_list=d_vars)
        g_trainer = self.optimization(self.g_lr).minimize(g_loss, var_list=g_vars)

        # From this point forward, reuse variables
        tf.get_variable_scope().reuse_variables()
        sess = tf.Session()
        if self.train:

            # Send summary statistics to TensorBoard
            """
            If you run this script on your own machine, include the cell below. Then, in a terminal window from the directory that this notebook lives in, run
            tensorboard --logdir=tensorboard/
            and open TensorBoard by visiting [`http://localhost:6006`](http://localhost:6006) in your web browser.
            """
            tf.summary.scalar('Generator_loss', g_loss)
            tf.summary.scalar('Discriminator_loss', d_loss)
            merged = tf.summary.merge_all()
            logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            writer = tf.summary.FileWriter(logdir, sess.graph)

            sess.run(tf.global_variables_initializer())

            try:
                # Train generator and discriminator together
                for i in range(self.iteration):
                    real_image_batch = next_batch(self.data, self.batch_size).reshape([self.batch_size, int(np.sqrt(self.x_dimensions)), int(np.sqrt(self.x_dimensions)), 1])
                    z_batch = self.generate_z(self.batch_size, 'None').reshape([self.batch_size, self.z_dimensions])

                    _ = sess.run(d_train, feed_dict={x_placeholder: real_image_batch, z_placeholder: z_batch,
                                                     global_: i})
                    # Train generator
                    _ = sess.run(g_trainer, feed_dict={x_placeholder: real_image_batch, z_placeholder: z_batch, global_: i})
                    if i % 10 == 0:
                        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
                        writer.add_summary(summary, i)
                print("Training has been completed")

            finally:
                # Optionally, uncomment the following lines to update the checkpoint files attached to the tutorial.
                saver = tf.train.Saver()
                saver.save(sess, self.model_path)
                print("Model has been saved to:" + self.model_path)
        if self.test:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)
            z_batch = self.generate_z(1, 'None').reshape([1, self.z_dimensions])
            z_placeholder = tf.placeholder(tf.float32, [1, self.z_dimensions], name='z_placeholder')
            generated_images = self.generator(tf.reshape(z_placeholder, [1, int(np.sqrt(self.z_dimensions)), int(np.sqrt(self.z_dimensions)), 1]), dim=self.x_dimensions, reuse=True)
            res = np.array(0)
            for i in range(self.test_n):
                images = sess.run(generated_images, {z_placeholder: z_batch})
                res_ = np.array(images)
                if res.all() == 0:
                    res = res_
                else:
                    res = np.vstack((res, res_))
            if self.test_image:
                # 可视化样本，下面是输出了训练集中前20个样本
                fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
                ax = ax.flatten()
                for i in range(20):
                    img = res[i].reshape(int(np.sqrt(self.x_dimensions)), int(np.sqrt(self.x_dimensions)))
                    ax[i].imshow(img, cmap='Greys')
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                plt.tight_layout()
                figdir = "fig/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
                plt.savefig(figdir)
                plt.show()

            res = pd.DataFrame(res)
            print("Dimensions of generating data：", res.shape)
            res.to_csv(self.test_path, index=None)