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

def time_linear(alpha=1.0):
    def linear_alpha(data, buy_date, not_util=False):
        flag = False
        if not_util and buy_date - data[-1, 3] <= 30:
            flag = True
        click_inf = 0.
        buy_inf   = 0.
        favo_inf  = 0.
        cart_inf  = 0.
        for i in data:
            if flag and buy_date - i[3] <= 30:
                if click_inf == 0. and buy_inf == 0. and favo_inf == 0. and cart_inf == 0.:
                    return None
                break
            inf = 1./(1+alpha*(buy_date - i[3]))
            if i[2] == 0:
                click_inf += inf
            elif i[2] == 1:
                buy_inf += inf
            elif i[2] == 2:
                favo_inf += inf
            elif i[2] == 3:
                cart_inf += inf
        if flag or not_util or buy_date - data[-1, 3] > 30:
            return np.array([click_inf, buy_inf, favo_inf, cart_inf, 0])
        else:
            return np.array([click_inf, buy_inf, favo_inf, cart_inf, 1])
    return linear_alpha

def extract_feature(data, kernel=time_linear()):
    sort_by(data)
    new_data = None
    for ui in np.unique(data[:, 0]):
        u_data = data[data[:, 0] == ui]
        for bi in np.unique(u_data[:, 1]):
            ub_data = u_data[u_data[:, 1] == bi]
            whether_buy = ub_data[:, 2] == 1
            i = 0
            while i < len(whether_buy):
                buy_ix = i + whether_buy[i:].argmax()
                if buy_ix == 0 and whether_buy[buy_ix]:
                    i += 1
                    continue
                if whether_buy[buy_ix]:
                    rec = kernel(ub_data[:buy_ix], ub_data[buy_ix, 3])
                    new_data = np.vstack((new_data, rec)) if new_data is not None else rec
                    i = buy_ix + 1
                else:
                    rec = kernel(ub_data, pre.BOUND, not_util=True)
                    if rec is not None:
                        new_data = np.vstack((new_data, rec)) if new_data is not None else rec
                    break
    return new_data

if __name__ == '__main__':
    data_path = os.path.join(
        os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
        'data', 'train_data.npy')
    train_data = np.load(data_path)
    inf_data = extract_feature(train_data, kernel=time_linear())
