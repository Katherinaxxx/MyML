#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/5/13 下午12:27
@Author : Catherinexxx
@Site : 
@File : clf.py
@Software: PyCharm
"""
import torch.nn as nn
import torch.optim as optim
import utils, torch, time, os, pickle
import numpy as np
from utils import load_data
from GAN import GAN
from CGAN import CGAN
from LSGAN import LSGAN
from DRAGAN import DRAGAN
from ACGAN import ACGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from infoGAN import infoGAN
from EBGAN import EBGAN
from BEGAN import BEGAN
from TOGAN import TOGAN


class classifier(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        # x = torch.cat([input, label], 1)
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class CLF(object):
    def __init__(self, args, fake_data=None):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.fake_model_name = args.gan_type
        self.model_name = args.clf_type
        self.use_fake_data = args.use_fake_data
        self.sample_num = args.fake_num
        self.input_size = args.input_size
        self.class_num = 10

        if self.dataset == 'cifar10':
            self.classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            self.classes = ('0', '1', '2', '3', '4', '5', '6', '7',
                            '8', '9')

        # load dataset
        # self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)

        if self.use_fake_data:
            self.data_loader = load_data(self.dataset, True, self.batch_size, True, fake_data)      # imbalanceed dataset

        else:
            self.data_loader = load_data(self.dataset, True, self.batch_size, True)      # imbalanceed dataset
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.net = classifier(input_dim=data.shape[1], output_dim=self.class_num, input_size=self.input_size, class_num=self.class_num)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrC, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.net.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.net)
        print('-----------------------------------------------')

    def train(self):
        self.train_hist = {}
        self.train_hist['Clf_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.net.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(x_)
                loss = self.criterion(outputs, y_)
                loss.backward()
                self.optimizer.step()
                self.train_hist['Clf_loss'].append(loss.item())

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d]  loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!")
        self.save()
        print("Saved training results!")
        with torch.no_grad():
            self.precisiom_recall()

    def precisiom_recall(self):
        self.net.eval()

        if self.use_fake_data:
            self.data_loader = load_data(self.dataset, True, self.batch_size, True)      # imbalanceed dataset

        correct = 0
        total = 0
        TP = list(0. for i in range(10))                        # TP
        actually_positive = list(0. for i in range(10))         # TP + FN （actually positive） y_
        predicted_positive = list(0. for i in range(10))        # TP + FP （predicted positive）predicted
        for iter, (x_, y_) in enumerate(self.data_loader):
            if iter == self.data_loader.dataset.__len__() // self.batch_size:
                break

            outputs = self.net(x_)

            _, predicted = torch.max(outputs.data, 1)
            total += y_.size(0)
            correct += (predicted == y_).sum().item()
            c = (predicted == y_).squeeze()         # TP

            for i in range(self.class_num):
                predicted_positive[i] += (predicted == i).sum().item()
                actually_positive[i] += (y_ == i).sum().item()
            for i in range(self.batch_size):
                label1 = y_[i]
                TP[label1] += c[i].item()

        print(correct, total)
        print(TP,actually_positive,predicted_positive)

        print('Accuracy of the network on the %d test images: %.4f %%' % (total,
                100 * correct / total))

        for i in range(self.class_num):
            print('%5s  Precision : %.4f %%, Recall : %.4f %%' % (
                self.classes[i], 100 * TP[i] / predicted_positive[i], 100 * TP[i] / actually_positive[i]))



    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.net.state_dict(), os.path.join(save_dir, self.model_name + '_clf.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.net.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_clf.pkl')))

        self.precisiom_recall()