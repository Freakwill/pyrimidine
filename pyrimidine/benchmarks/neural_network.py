#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA

def relu(x):
    return (x > 0) * x

def _mlp(X, A1, b1, A2, b2):
    N = X.shape[0]
    O1 = relu(np.dot(X, A1)+ np.tile(b1, (N, 1)))
    return np.dot(O1, A2)+ np.tile(b2, (N, 1))

class MLP:
    """MLP

    Y = A2f(A1X+b1)+b2
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    @staticmethod
    def random(N=100, p=2):
        X = np.random.random(size=(N, p))
        Y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.sin(X[:, 0]*2) + np.cos(X[:, 1]*2)

        return MLP(X, Y)


    def __call__(self, x):
        E = LA.norm(_mlp(self.X, *x) - self.Y) / LA.norm(self.Y)
        return -E
