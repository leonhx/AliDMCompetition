#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import re
import os

YEAR = 2013
current_dir = os.path.split(os.path.abspath(__file__))[0]
csv_data_path = os.path.join(current_dir, 't_alibaba_data.csv')
raw_data_path = os.path.join(os.path.join(current_dir, 'raw_data.npy'))

def date2int(date):
    START_DATE = datetime.date(YEAR, 1, 1)
    return (date - START_DATE).days

def date_parser(s):
    DATE_RE = re.compile(u'(\d+)月(\d+)日')
    m = DATE_RE.match(s.decode('gbk'))
    return date2int(datetime.date(YEAR, int(m.group(1)), int(m.group(2))))

if __name__ == '__main__':
    raw_data = np.loadtxt(csv_data_path, dtype=int, delimiter=',', converters={3: date_parser}, skiprows=1)
    np.save(raw_data_path, raw_data)
