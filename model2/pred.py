#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib

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

def time_linear(alpha=1.0):
    def linear_alpha(data, end_date):
        click_inf = 0.
        buy_inf   = 0.
        favo_inf  = 0.
        cart_inf  = 0.
        for i in data:
            inf = 1./(1+alpha*(end_date - i[3]))
            if i[2] == 0:
                click_inf += inf
            elif i[2] == 1:
                buy_inf += inf
            elif i[2] == 2:
                favo_inf += inf
            elif i[2] == 3:
                cart_inf += inf
        return np.array([click_inf, buy_inf, favo_inf, cart_inf])
    return linear_alpha

def use_kernel(kernel, data, buy_date, not_util=False):
    y = 1
    if not_util:
        data = data[data[:, 3] < buy_date - 30]
        if data.shape[0] == 0:
            return None
    if not_util or buy_date - data[-1, 3] > 30:
        y = 0
    return kernel(data, buy_date), y

def get_instances(ub_data, kernel):
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

def extract_feature(data, kernel=time_linear()):
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

if __name__ == '__main__':
    data_path = os.path.join(
        os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
        'data', 'train_data.npy')
    train_data = np.load(data_path)
    inf_data, y = extract_feature(train_data, kernel=time_linear())

    param_grid = [
        {'C': [1, 10, 100], 'kernel': ['linear']},
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        {'C': [1, 10, 100], 'degree': [2, 3], 'kernel': ['poly']},
    ]
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring='f1', n_jobs=-1)
    clf.fit(inf_data, y)

    print('Best parameters set found on development set: %s' % clf.best_estimator_)
    print('Best f1 score: %s' % clf.best_score_)
