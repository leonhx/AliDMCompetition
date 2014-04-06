#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
A script to validate model automatically or generate result file of the model.

Usage
-----
    $ python run.py val model-sgm model-lr model-lfm
    $ python run.py gen model-sgm

Arguments
---------

1:  command, 'val' or 'gen'.
    'val' - validate the model
    'gen'  - generate result file of the model

2~: model name, i.e., the folder where corresponding pred.py lies
    *NOTE*: command `gen` accepts one model only

Note
----

There should be a function named `get_model` in pred.py, which returns
a properly initialized model object.

A model object should have at least two methods: `fit` and `predict`.
`fit`: ndarray(column=[user_id, brand_id, type, visit_datetime]) -> None
`predict`: int (time now) -> (ndarray(column=[user_id, item_id]), ndarray(column = [rating]))
"""

import numpy as np
import pylab as pl

import sys
import os
import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data')
sys.path.append(data_path)
import prep

all_data = np.load(os.path.join(data_path, 'raw_data.npy'))

def ndarray2dict(ndarray):
    result = {}
    for u, i in ndarray:
        result.setdefault(u, set())
        result[u].add(i)
    return result

def dict_size(dictionary):
    size = 0
    for i in dictionary:
        size += len(dictionary[i])
    return size

def f1(pred_result, val_result, all_userids):
    """
    pred_result has the same format as val_result:
        key: user_id
        value: set of brand_id
    """
    pBrands = sum([len(pred_result[ui]) for ui in pred_result])
    hitBrands = 0
    for ui in all_userids:
        if ui in pred_result and ui in val_result:
            hitBrands += len(pred_result[ui].intersection(val_result[ui]))
    bBrands = sum([len(val_result[ui]) for ui in val_result])
    p = hitBrands * 1. / pBrands
    assert 0 < p <= 1
    r = hitBrands * 1. / bBrands
    assert 0 < r <= 1
    return p, r, 2.*p*r/(p+r)

def stats(result, data):
    """
    result: dict
        key: user_id
        value: set of brand_id
    data: ndarray, column=[user_id, brand_id, type, visit_datetime]
    """
    total = dict_size(result)
    visit = bought = favo = cart = new = 0
    for ui in result:
        u_result = result[ui]
        u_data = data[data[:, 0] == ui]
        uv_brands = set(u_data[u_data[:, 2] == 0, 1])
        ub_brands = set(u_data[u_data[:, 2] == 1, 1])
        uf_brands = set(u_data[u_data[:, 2] == 2, 1])
        uc_brands = set(u_data[u_data[:, 2] == 3, 1])
        visit += len(u_result.intersection(uv_brands))
        bought += len(u_result.intersection(ub_brands))
        favo += len(u_result.intersection(uf_brands))
        cart += len(u_result.intersection(uc_brands))
        new += len(u_result.difference(set(u_data[:, 1])))
    return total, visit, bought, favo, cart, new

def print_model_header(model_name):
    print('======')
    print('Model:\t%s' % model_name)

def print_result_header(description):
    print('****** %s' % description)

def print_basic_result(p, r, f):
    print('Precision: {:f}%'.format(p*100))
    print('Recall:    {:f}%'.format(r*100))
    print('F1 Score:  {:f}%'.format(f*100))

def print_result_stats(pred_stats, real_stats, p):
    print('|         TOTAL   VISITED BOUGHT  FAVO    CART    NEW')
    print('| Pred #  {:<8}{:<8}{:<8}{:<8}{:<8}'.format(*pred_stats))
    print('|      %  {:<8.0%}{:<8.3%}{:<8.3%}{:<8.3%}{:<8.3%}'.format(*[i*1./pred_stats[0] for i in pred_stats]))
    print('| Real #  {:<8}{:<8}{:<8}{:<8}{:<8}'.format(*real_stats))
    print('|      %  {:<8.0%}{:<8.3%}{:<8.3%}{:<8.3%}{:<8.3%}'.format(*[i*1./real_stats[0] for i in real_stats]))
    print('#Hit:  %d' % round(pred_stats[0]*p))

def plot_result(model_name, val_cases, Ps, Rs, Fs):
    pl.figure(model_name)
    x = range(len(val_cases))
    pl.plot(x, Ps)
    pl.plot(x, Rs)
    pl.plot(x, Fs)
    pl.legend(('Precision', 'Recall', 'F1 Score'))
    pl.xticks(x, [i[3] for i in val_cases])
    pl.show()

def get_pred(model, data, bound_date):
    model.fit(data)
    predictions, _ = model.predict(bound_date)
    return ndarray2dict(predictions)

def get_val(val_data, val_userids):
    raw_val_data = val_data[val_data[:, 2] == 1]
    real_bought = []
    for u, b, _, _ in raw_val_data:
        if u in val_userids:
            real_bought.append([u, b])
    return ndarray2dict(real_bought)

def val():
    if len(sys.argv) < 3:
        raise LookupError('Please specify model to val.')
    val_cases = [
        (prep.date(5, 18), prep.date(6, 17), 'predicts 5/18 to 6/17', '5/18'),
        (prep.date(6, 17), prep.date(7, 17), 'predicts 6/17 to 7/17', '6/17'),
        (prep.date(7, 17), prep.date(8, 16), 'predicts 7/17 to 8/16', '7/17'),
    ]
    for model_name in sys.argv[2:]:
        sys_path = sys.path[:]
        model_path = os.path.join(current_dir, model_name)
        sys.path.append(model_path)
        import pred
        reload(pred)
        print_model_header(model_name)
        Ps = []
        Rs = []
        Fs = []
        for TRAIN_DATE, VAL_DATE, DESC, _ in val_cases:
            train_data = all_data[all_data[:, 3] < TRAIN_DATE]
            val_data = all_data[np.logical_and(
                all_data[:, 3] >= TRAIN_DATE,
                all_data[:, 3] < VAL_DATE
            )]
            all_userids = np.unique(train_data[:, 0])
            pred_result = get_pred(pred.get_model(), train_data, TRAIN_DATE-1)
            val_result = get_val(val_data, all_userids)
            p, r, f = f1(pred_result, val_result, all_userids)
            Ps.append(p)
            Rs.append(r)
            Fs.append(f)
            print_result_header(DESC)
            print_result_stats(stats(pred_result, train_data),
                stats(val_result, train_data), p)
            print_basic_result(p, r, f)
        plot_result(model_name, val_cases, Ps, Rs, Fs)
        sys.path = sys_path

def output(filename, pred_result):
    lines = []
    for u, items in pred_result.items():
        line = '{0}\t{1}\n'.format(u, ','.join([str(i) for i in items]))
        lines.append(line)
    f = open(filename, 'w')
    f.writelines(lines)
    f.close()

def gen():
    sys_path = sys.path[:]
    if len(sys.argv) < 3:
        raise LookupError('Please specify model result to transfer.')
    elif len(sys.argv) > 3:
        raise UserWarning('Only one model result can be transfered.')
    model_path = os.path.join(current_dir, sys.argv[2])
    sys.path.append(model_path)
    import pred
    pred_result = get_pred(pred.get_model(), all_data, prep.date(8, 15))
    pred_stats = stats(pred_result, all_data)
    print('|         TOTAL   VISITED BOUGHT  FAVO    CART    NEW')
    print('| Pred #  {:<8}{:<8}{:<8}{:<8}{:<8}'.format(*pred_stats))
    print('|      %  {:<8.0%}{:<8.3%}{:<8.3%}{:<8.3%}{:<8.3%}'.format(*[i*1./pred_stats[0] for i in pred_stats]))
    from datetime import date
    target_path = os.path.join(current_dir, 'result%02d.txt'%date.today().day)
    output(target_path, pred_result)
    sys.path = sys_path

if __name__ == '__main__':
    if sys.argv[1] not in ['val', 'gen']:
        raise LookupError('Unknown command: %d' % sys.argv[1])
    eval('%s()' % sys.argv[1])
