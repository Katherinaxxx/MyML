#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/9/9 下午7:43
@Author : Catherinexxx
@Site : 
@File : XGBoost.py
@Software: PyCharm
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score     # K折交叉验证模块
from xgboost.sklearn import XGBClassifier
from sklearn.externals import joblib                    # jbolib模块


class XGBoost(object):
    def __init__(self, trainx, trainy, testx, testy, predictx, ModelPath='model/xgb_clf.pkl', train=True, predict=True,
                 silent=0, nthread=4, learning_rate=0.3, min_child_weight=1, max_depth=6, gamma=0, subsample=1,
                 max_delta_step=0, colsample_bytree=1, reg_lambda=1, reg_alpha=0, scale_pos_weight=1,
                 objective='multi:softmax', num_class=10, n_estimators=100, seed=1000, eval_metric='auc'
                 ):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.predictx = predictx
        self.ModelPath = ModelPath
        self.silent = silent                        # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        self.nthread = nthread                      # cpu 线程数 默认最大
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        self.max_depth = max_depth                  # 构建树的深度，越大越容易过拟合
        self.gamma = gamma                          # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        self.subsample = subsample                  # 随机采样训练样本 训练实例的子采样比
        self.max_delta_step = max_delta_step        # 最大增量步长，我们允许每个树的权重估计。

        self.colsample_bytree = colsample_bytree    # 生成树时进行的列采样

        self.reg_lambda = reg_lambda                # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        self.reg_alpha = reg_alpha                  # L1 正则项参数
        self.scale_pos_weight = scale_pos_weight    # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重

        self.objective = objective                  # 多分类的问题 指定学习任务和相应的学习目标
        """
        binary:logistic-用于二分类，返回分类的概率而不是类别（class）
        multi:softmax-多分类问题，返回分类的类别而不是概率
        multi:softprob-与softmax类似，但是返回样本属于每一类的概率
        """
        self.num_class = num_class                  # 类别数，多分类与 multisoftmax 并用

        self.n_estimators = n_estimators
        self.seed = seed
        self.eval_metric = eval_metric

    def model(self):
        clf = XGBClassifier(silent=self.silent, nthread=self.nthread, learning_rate=self.learning_rate,
                            min_child_weight=self.min_child_weight, max_depth=self.max_depth, gamma=self.gamma,
                            subsample=self.subsample, max_delta_step=self.max_delta_step, colsample_bytree=self.colsample_bytree,
                            reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, scale_pos_weight=self.scale_pos_weight,
                            objective=self.objective, num_class=self.num_class, n_estimators=self.n_estimators, seed=
                            self.seed, eval_metric=self.eval_metric
                            )
        return clf

    def train(self):
        clf = self.model()
        clf.fit(self.trainx, self.trainy, eval_metric='auc')
        print("AUC of training set:", clf.score(self.trainx, self.trainy))
        # CV
        scores = cross_val_score(clf, self.testx, self.testy, cv=5, scoring='accuracy')
        print("Accuracy (CV):", scores)
        print("Accuracy (mean):", scores.mean())  # 0.9290273587379518
        # save model
        joblib.dump(clf, self.ModelPath)
        print("Model has been saved to ", self.ModelPath)

    def predict(self):
        # restore model
        clf = joblib.load('model/xgb_clf.pkl')
        # predict test set
        preds = clf.predict_proba(self.predictx)[:, 1]
        preds = pd.DataFrame([x for x in preds])
        preds = MinMaxScaler().fit_transform(preds.reshape(-1, 1))
        return preds
