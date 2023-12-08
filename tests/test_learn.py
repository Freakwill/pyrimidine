#!/usr/bin/env python3


import numpy as np

from pyrimidine.learn.neural_network import GAANN
from pyrimidine.learn.linear_regression import GALinearRegression


def test_ann():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([[0,1], [1,0], [1,0], [0,1]])

    model = GAANN()
    pop = GAANN.config(X, Y)
    s0 = pop.fitness
    model.max_iter = 2
    model.fit(X, Y, pop)
    s1 = model.score(X, Y)
    assert s0 <= s1

def test_lr():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([2,1,0,0])

    model = GALinearRegression()
    pop = GALinearRegression.config(X, Y)
    for i in pop:
        i._fitness()
    s0 = pop.fitness
    model.max_iter = 2
    model.fit(X, Y, pop)
    s1 = model.score(X, Y)
    assert s0 <= s1
