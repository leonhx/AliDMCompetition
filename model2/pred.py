#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os
sys.path.append(
    os.path.join(
        os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
        'data'))
import preprocess as pre

def sort_by(data, order=['user_id', 'brand_id', 'visit_datetime']):
    actype = np.dtype({
        'names': ['user_id', 'brand_id', 'type', 'visit_datetime'],
        'formats': [np.long, np.long, np.int, np.int]
        })
    typed_data = np.zeros(len(data), dtype=actype)
    for i in range(len(data)):
        typed_data[i] = tuple(data[i])
    typed_data.sort(order=order)
    for i in range(len(typed_data)):
        td = typed_data[i]
        data[i][0] = td['user_id']
        data[i][1] = td['brand_id']
        data[i][2] = td['type']
        data[i][3] = td['visit_datetime']

def time_poly(alpha=1.0, n=2):
    def poly(data, end_date):
        click_inf = 0.
        buy_inf   = 0.
        favo_inf  = 0.
        cart_inf  = 0.
        for i in data:
            inf = 1./(1+alpha*(end_date - i[3])**n)
            if i[2] == 0:
                click_inf += inf
            elif i[2] == 1:
                buy_inf += inf
            elif i[2] == 2:
                favo_inf += inf
            elif i[2] == 3:
                cart_inf += inf
        return np.array([click_inf, buy_inf, favo_inf, cart_inf])
    return poly

def use_kernel(kernel, data, buy_date, not_util=False):
    y = 1
    if not_util:
        data = data[data[:, 3] < buy_date - 30]
        if data.shape[0] == 0:
            return None
    if not_util or buy_date - data[-1, 3] > 30:
        y = 0
    return kernel(data, buy_date), y

def get_train_instances(ub_data, kernel):
    whether_buy = ub_data[:, 2] == 1
    xs = []
    ys = []
    i = 0
    while i < len(whether_buy):
        buy_ix = i + whether_buy[i:].argmax()
        if buy_ix == 0 and whether_buy[buy_ix]:
            i += 1
            continue
        if whether_buy[buy_ix]:
            x, y = use_kernel(kernel, ub_data[:buy_ix], ub_data[buy_ix, 3])
            xs.append(x)
            ys.append(y)
            i = buy_ix + 1
        else:
            rec = use_kernel(kernel, ub_data, pre.BOUND, not_util=True)
            if rec is not None:
                x, y = rec
                xs.append(x)
                ys.append(y)
            break
    return xs, ys

def get_pred_instance(ub_data, kernel):
    return [kernel(ub_data, pre.BOUND)], [np.array([ub_data[0, 0], ub_data[0, 1]])]

def extract_feature(data, kernel, get_instances):
    sort_by(data)
    xs = []
    ys = []
    for ui in np.unique(data[:, 0]):
        u_data = data[data[:, 0] == ui]
        for bi in np.unique(u_data[:, 1]):
            ub_data = u_data[u_data[:, 1] == bi]
            xs_, ys_ = get_instances(ub_data, kernel)
            xs += xs_
            ys += ys_
    return np.array(xs), np.array(ys)

class AdaBoost:
    def __init__(self, clf_list):
        self.__clfs__ = clf_list
    def predict(self, X):
        ys = [clf.predict(X) for clf in self.__clfs__]
        y = sum(ys)
        y[y <= len(self.__clfs__)/2] = 0
        y[y > len(self.__clfs__)/2] = 1
        return y
    def fit(X, y):
        _ = [clf.fit(X, y) for clf in self.__clfs__]

def ada_boost(clf_list):
    return AdaBoost(clf_list)

if __name__ == '__main__':
    data_path = os.path.join(
        os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
        'data', 'train_data.npy')
    data = np.load(data_path)
    poly_kernel = time_poly(alpha=0.5, n=1)

    X, y = extract_feature(data, poly_kernel, get_train_instances)

    from sklearn.svm import LinearSVC
    svc = LinearSVC(C=10, loss='l1')

    from sklearn.linear_model import LogisticRegression, Perceptron, PassiveAggressiveClassifier
    lr = LogisticRegression(penalty='l1')
    pc = Perceptron(penalty='l1')
    pa = PassiveAggressiveClassifier(C=1, loss='hinge')

    clf = ada_boost([svc, lr, pc, pa])
    clf.fit(X, y)

    # from sklearn.externals import joblib
    # clf = joblib.load(
    #     os.path.join(
    #         os.path.split(os.path.abspath(__file__))[0],
    #         'rbf.C=10.gamma=0.001.svc'))
    pred_X, ub = extract_feature(data, poly_kernel, get_pred_instance)
    y = clf.predict(pred_X)
    ub = ub[y == 1]

    pred_result = {}
    for ui, bi in ub:
        pred_result.setdefault(ui, set())
        pred_result[ui].add(bi)
    import pickle
    f = open(
        os.path.join(
            os.path.split(os.path.abspath(__file__))[0], 'pred_result.pkl'),
        'w')
    pickle.dump(pred_result, f)
    f.close()
