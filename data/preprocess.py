#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import re

YEAR = 2013
START_DATE = datetime.date(YEAR, 1, 1)
date_re = re.compile(u'(\d+)月(\d+)日')

def date_parser(s):
    m = date_re.match(s.decode('gbk'))
    date_delta = datetime.date(YEAR, int(m.group(1)), int(m.group(2))) - START_DATE
    return date_delta.days

raw_data = np.loadtxt('t_alibaba_data.csv', dtype=int, delimiter=',', converters={3: date_parser}, skiprows=1)

bound = (datetime.date(YEAR, 7, 16) - START_DATE).days

train_data = raw_data[raw_data[:, 3] <= bound]
test_data = raw_data[raw_data[:, 3] > bound]
t1_test = test_data[test_data[:, 2] == 1, :2]

test_result = {}
for ui, bi in t1_test:
    test_result.setdefault(ui, set())
    test_result[ui].add(bi)

np.save('raw_data', raw_data)
np.save('train_data', train_data)
np.save('test_data', test_data)
import pickle
f = open('test_result.pkl', 'w')
pickle.dump(test_result, f)
f.close()
