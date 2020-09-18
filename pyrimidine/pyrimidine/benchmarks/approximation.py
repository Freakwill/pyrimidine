#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.linalg as LA


_basis = [lambda x: np.ones(len(x)), lambda x: x, lambda x: x**2,
np.sin, np.cos, np.tan,
np.exp, lambda x: np.log(np.abs(x)+1)]

n_basis_ = len(_basis)

class Function1DApproximation:
    def __init__(self, function, lb=0, ub=1):
        self.function = function
        self.lb = lb
        self.ub = ub
        self.x = np.linspace(self.lb, self. ub, 100)
        self.y = self.function(self.x)
        self.threshold = 1


    def __call__(self, coefs):
        """A: N * K
        C: K
        B: K*p
        """
        yy = np.sum([c*b(self.x) for c, b in zip(coefs, _basis)], axis=0)
        epsilon = 0.01
        return - np.mean(np.abs((yy + epsilon) / (self.y + epsilon) - 1)) + np.mean(np.abs(coefs) < self.threshold)
