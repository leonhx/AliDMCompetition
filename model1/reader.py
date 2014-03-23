#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series
import numpy as np
import datetime
import re

YEAR = 2013
START_DATE = datetime.date(YEAR, 1, 1)
date_re = re.compile(u'(\d+)月(\d+)日')

def date_parser(s):
    m = date_re.match(s.decode('gbk'))
    return datetime.date(YEAR, int(m.group(1)), int(m.group(2)))

raw_data = pd.read_csv('t_alibaba_data.csv', parse_dates=[3], date_parser=date_parser)
scores = np.where(raw_data.ix[:, 2] == 1, 4, np.where(raw_data.ix[:, 2] == 0, 1, raw_data.ix[:, 2]))
raw_data['score'] = Series(scores)
raw_data['score'] = raw_data['type']

# test_data = raw_data.ix[raw_data.visit_datetime > datetime.date(YEAR, 7, 15), :]
train_data = raw_data.ix[raw_data.visit_datetime <= datetime.date(YEAR, 7, 15), :]

# test_behaviour = test_data.ix[test_data.score == 4, :][['user_id', 'brand_id']]
# test_behaviour = test_data.ix[test_data.score == 1, :][['user_id', 'brand_id']]
# test_result = {}
# for r in test_behaviour.iterrows():
#     ui, bi = r[1].user_id, r[1].brand_id
#     test_result.setdefault(ui, set())
#     test_result[ui].add(bi)

import pickle

# f = open('raw_data.pkl', 'w')
# pickle.dump(raw_data, f)
# f.close()
# f = open('train_data.pkl', 'w')
# pickle.dump(train_data, f)
# f.close()
# f = open('test_result.pkl', 'w')
# pickle.dump(test_result, f)
# f.close()

# f = open('raw_data.pkl', 'r')
# raw_data = pickle.load(f)
# f.close()
# f = open('train_data.pkl', 'r')
# train_data = pickle.load(f)
# f.close()
f = open('test_result.pkl', 'r')
test_result = pickle.load(f)
f.close()
