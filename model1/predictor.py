#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath('.'))

import reader

train_data = reader.train_data

class ScoreVec:
    def __init__(self):
        self.__vec__ = {}
        self.__size__ = None
    def __getitem__(self, brand_id):
        return self.__vec__.get(brand_id, 0)
    def __setitem__(self, brand_id, score):
        self.__vec__.setdefault(brand_id, 0.)
        self.__vec__[brand_id] = max(score, self.__vec__[brand_id])
        self.__size__ = None
    def size(self):
        if not self.__size__:
            import math
            self.__size__ = math.sqrt(sum([x**2 for x in self.__vec__.values()]))
        return self.__size__
    def __mul__(self, other):
        self_i = self.__vec__.keys()
        other_i = other.__vec__.keys()
        return sum([self.__vec__[i]*other.__vec__[i] for i in set(self_i).intersection(other_i)])
    def brands(self):
        return self.__vec__.keys()
    def similarity(self, other):
        return self * other * 1.0 / (self.size() * other.size())

class ScoreGridIndex:
    def __init__(self, sg):
        self.__sg__ = sg
    def __getitem__(self, ij):
        i, j = ij
        users = self.__sg__.users()
        return self.__sg__[users[i], users[j]]

class ScoreGrid:
    def __init__(self):
        self.__vec__ = {}
        self.__users__ = None
        self.ix = ScoreGridIndex(self)
    def __getitem__(self, ij):
        if type(ij) != tuple:
            return self.__vec__.get(ij)
        i, j = ij
        if i == j:
            return 1.
        if i not in self.__vec__:
            return None
        return self.__vec__.get(i).get(j, 0.)
    def __setitem__(self, ij, score):
        i, j = ij
        self.__vec__.setdefault(i, {})
        self.__vec__.setdefault(j, {})
        self.__vec__[i][j] = score
        self.__vec__[j][i] = score
        self.__users__ = None
    def users(self):
        if not self.__users__:
            self.__users__ = self.__vec__.keys()
        return self.__users__
    def maximum(self, user_id):
        highest = 0.
        other = None
        for ui in self.__vec__[user_id]:
            if self.__vec__[user_id][ui] > highest:
                highest = self.__vec__[user_id][ui]
                other = ui
        return other
    def minimum(self, user_id):
        lowest = 0.
        other = None
        for ui in self.__vec__[user_id]:
            if self.__vec__[user_id][ui] > lowest:
                lowest = self.__vec__[user_id][ui]
                other = ui
        return other

import pickle

# users = {}
# for r in train_data.iterrows():
#     users.setdefault(r[1].user_id, ScoreVec())
#     users[r[1].user_id][r[1].brand_id] = r[1].score

# f = open('users.pkl', 'w')
# pickle.dump(users, f)
# f.close()

f = open('users.pkl', 'r')
users = pickle.load(f)
f.close()

# weights = ScoreGrid()
# for i in users:
#     for j in users:
#         if i < j:
#             weights[i, j] = users[i].similarity(users[j])

# f = open('weights.pkl', 'w')
# pickle.dump(weights, f)
# f.close()

f = open('weights.pkl', 'r')
weights = pickle.load(f)
f.close()

def predict(user_id):
    score = users[user_id]
    brand_ids = []
    picked_users = []
    for ui in weights[user_id]:
        if weights[user_id, ui] > 0.1:
            picked_users.append(ui)
    for bi in score.brands():
        bi_sc = 0.
        weight_sum = 0.
        for ui in picked_users:
            if ui != user_id and bi in users[ui].brands():
                bi_sc += users[ui][bi] * weights[user_id, ui]
                weight_sum += weights[user_id, ui]
        if weight_sum != 0.:
            if bi_sc / weight_sum > 2:
                brand_ids.append(bi)
    return brand_ids

pred_result = {}
for ui in users:
    pred_result.setdefault(ui, set())
    pred_result[ui].update(predict(ui))
