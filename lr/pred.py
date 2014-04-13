#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(data_path)
import prep

class LR:
    """
    Parameters
    ----------
    model: An object which has interfaces `fit' and `predict'

    alpha: float, optional(default=1.0)
        Penalty factor with respect to time

    degree: int, optional(default=2)
        Penalty degree on time interval

    """
    def __init__(self, model, alpha=1.0, degree=2):
        self.__poly_kernel__ = time_poly(alpha=alpha, n=degree)
        self.__model__ = model
    def fit(self, X):
        self.__data__ = X
    def predict(self, time_now):
        X, y = extract_feature(self.__data__, self.__poly_kernel__, get_train_instances, time_now)
        p_X = X[y == 1]
        n_X = X[y == 0]
        new_nX = []
        p = y.sum()*1. / len(y)
        n = 1. - p
        ratio = p / n
        from random import random
        for line in n_X:
            if random() < ratio*1.5:
                new_nX.append(line)
        new_nX = np.array(new_nX)
        X = np.r_[p_X, new_nX]
        y = np.array([1]*len(p_X) + [0]*len(new_nX), dtype=int)
        print len(new_nX) * 1. / len(p_X)
        self.__model__.fit(X, y)
        pred_X, ub = extract_feature(self.__data__, self.__poly_kernel__, get_pred_instance, time_now)
        y = self.__model__.predict(pred_X)
        predictions = ub[y == 1]
        return predictions, np.ones((len(predictions,)))

def get_model():
    from sklearn.svm import LinearSVC
    return LR(model=LinearSVC(C=10000, loss='l1'), alpha=0.7, degree=1)

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
    def polyi(data, end_date):
        vec = []
        intervals = [0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 40, 45,
            50, 60, 70, 80, 90, 10000]
        for i in range(len(intervals)-2):
            click_inf = 0.
            buy_inf   = 0.
            favo_inf  = 0.
            cart_inf  = 0.
            start = end_date - intervals[i]
            end = end_date - intervals[i+1]
            for i in data[np.logical_and(data[:, 3]>end, data[:, 3]<=start)]:
                inf = 1./(1+alpha*(start - i[3])**n)
                if i[2] == 0:
                    click_inf += inf
                elif i[2] == 1:
                    buy_inf += inf
                elif i[2] == 2:
                    favo_inf += inf
                elif i[2] == 3:
                    cart_inf += inf
            vec += [click_inf, buy_inf, favo_inf, cart_inf]
        return np.array(vec)
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

def use_kernel(kernel, data, bound_date, not_util=False):
    y = 1
    if not_util:
        data = data[data[:, 3] < bound_date - 30]
        if data.shape[0] == 0:
            return None
    if not_util or bound_date - data[-1, 3] > 30:
        y = 0
    return kernel(data, bound_date), y

def get_train_instances(ub_data, kernel, bound_date):
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
            rec = use_kernel(kernel, ub_data, bound_date, not_util=True)
            if rec is not None:
                x, y = rec
                xs.append(x)
                ys.append(y)
            break
    return xs, ys

def get_pred_instance(ub_data, kernel, bound_date):
    return [kernel(ub_data, bound_date)], [np.array([ub_data[0, 0], ub_data[0, 1]])]

def extract_feature(data, kernel, get_instances, bound_date):
    sort_by(data)
    xs = []
    ys = []
    for ui in np.unique(data[:, 0]):
        u_data = data[data[:, 0] == ui]
        for bi in np.unique(u_data[:, 1]):
            ub_data = u_data[u_data[:, 1] == bi]
            xs_, ys_ = get_instances(ub_data, kernel, bound_date)
            xs += xs_
            ys += ys_
    return np.array(xs), np.array(ys)
