#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(data_path)
import prep

class LPM:
    """
    Latent Pattern Model

    Parameters
    ----------
    """
    def __init__(self):
        pass
    def fit(self, X):
        pass
    def predict(self, time_now):
        pass

def get_model():
    from sklearn.svm import LinearSVC
    return LR(model=LinearSVC(C=10, loss='l1'), alpha=0.7, degree=1)
