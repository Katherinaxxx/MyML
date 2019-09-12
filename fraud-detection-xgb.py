#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/9/9 下午7:02
@Author : Catherinexxx
@Site : 
@File : fraud-detection-xgb.py
@Software: PyCharm
"""
import pandas as pd
from lib.classifiers.XGBoost import XGBoost
from sklearn.model_selection import train_test_split


trainx = pd.read_csv('datasets/ieee-fraud-detection/train.csv')
testx = pd.read_csv('datasets/ieee-fraud-detection/test.csv')


trainy = trainx.y
trainx = trainx.drop(['y'], axis=1)

print(trainx.shape)

train_x, test_x, train_y, test_y = train_test_split(trainx, trainy, test_size=0.3)

xgb = XGBoost(train_x, train_y, test_x, test_y, testx)
xgb.train()
pred = pd.DataFrame(xgb.predict())
pred.to_csv('datasets/ieee-fraud-detection/xgb_pred.csv')


