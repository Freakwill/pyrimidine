#!/usr/bin/env python3


import numpy as np
import numpy.linalg as LA

from ..benchmarks import BaseProblem

_basis = [lambda x: np.ones(len(x)), lambda x: x, lambda x: x**2,
np.sin, np.cos, np.tan,
np.exp, lambda x: np.log(np.abs(x)+1), lambda x:x>0]

n_basis_ = len(_basis)


def lin_comb(x, coefs, basis):
    return np.sum([c*b(x) for c, b in zip(coefs, basis)], axis=0)


class Function1DApproximation(BaseProblem):

    def __init__(self, function, lb=0, ub=1, basis=_basis):
        self.function = function
        self.lb = lb
        self.ub = ub
        self.x = np.linspace(self.lb, self. ub, 30)
        self.y = self.function(self.x)
        self.threshold = 1
        self.basis = basis

    def __call__(self, coefs):
        """A: N * K
        C: K
        B: K*p
        """

        yy = np.sum([c*b(self.x) for c, b in zip(coefs, self.basis)], axis=0)
        p = 0.001
        return - np.mean(np.abs((yy - self.y) / (np.abs(self.y) + 1))) + p*np.mean(np.abs(coefs) < self.threshold)
