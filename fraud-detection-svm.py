#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/9/5 下午8:09
@Author : Catherinexxx
@Site : 
@File : fraud-detection-svm.py
@Software: PyCharm
"""
from __future__ import division
import numpy as np
from SVM import *
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from sklearn import svm
from sklearn.metrics import  roc_curve

if sys.version_info[0] >= 3:
    xrange = range


class SVM(object):
    """
    one=True 用同一个dataset做svm 训练&验证
    one=False 用amplified+real训练 用real验证

    """
    def __init__(self, real_path, path='', one=True):
        self.path = path
        self.real_path = real_path
        self.one = one

    def import_data(self):
        df = pd.read_csv(self.real_path)
        df = np.array(df)
        df_x = df[:, :df.shape[1] - 2]
        df_y = df[:, df.shape[1] - 1]
        if self.one:
            train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3)
            return train_x, test_x, train_y, test_y
        else:
            df2 = pd.read_csv(self.path)
            df2 = np.array(df2)
            df_x2 = df2[:, :df2.shape[1]-2]
            df_y2 = df2[:, df2.shape[1]-1]

            return df_x, df_x2, df_y, df_y2

    def svm2class(self):
        train_x, test_x, train_y, test_y = self.import_data()
        C = 20
        toler = 0.001
        maxIter = 5000
        kernelOption = ("rbf", 20)
        svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption)

        print("Step3: Testing classifier......")
        accuracy, labelpredict, num = testSVM(svmClassifier, test_x, test_y)
        print("\tAccuracy = %.2f%%" % (accuracy * 100))

    def cal_rate(self, true, predict):
        sum = np.sum(true)
        mul = true * predict
        tmp = np.sum(np.where(mul == 1, 1, 0))
        accurate = np.sum(np.where(true == predict, 1, 0))
        accurate_rate = accurate / len(true)
        tpr = tmp / sum
        return tpr, accurate_rate

    def main(self):
        train_x, test_x, train_y, test_y = self.import_data()

        # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object
        model = svm.SVC(kernel='rbf', C=1, gamma=1)

        # there is various option associated with it, like changing kernel, gamma and C value. Will discuss more
        #  about it in next section.Train the model using the training sets and check score
        model.fit(train_x, train_y)
        model.score(train_x, train_y)
        # Predict Output
        predicted = model.predict(test_x)
        tpr, accurate = self.cal_rate(test_y, predicted)
        print('True Positive Rate : ', tpr)
        print('Accurate Rate : ', accurate)


if __name__ == '__main__':
    print('增强前')
    # SVM('logistic.csv').main()
    SVM('data_processed.csv').main()
    print('增强后')
    SVM('logistic.csv', 'syn_real.csv', False).main()     # 0.1
    SVM('data_processed.csv', 'syn_real.csv', False).main()     # 0.1
    # SVM('syn_amplified.csv', 'syn_real.csv').main()            # 0
    # SVM('/Users/xyh/Desktop/GAN/amplify-data/test34/logistic.csv').main()