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
        Y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.sin(X[:, 0]*2) + np.cos(X[:, 1]*2) * np.cos(X[:, 2])

        return MLP(X, Y)


    def __call__(self, x):
        E = LA.norm(_mlp(self.X, *x) - self.Y) / LA.norm(self.Y)
        return -E

def _rnn(X, A1, b1, A2, b2, C1, c1, C2, c2, Z=0):
    N = X.shape[0]
    if Z == 0:
        H = 1
    else:
        H = Z.shape[0]
    O1 = relu(np.dot(X, A1)+ np.tile(b1, (N, 1)) + np.dot(C1, Z))
    return np.dot(O1, A2)+ np.tile(b2, (N, 1)), np.dot(O1, C2)+ np.tile(c2, (H, 1))

class RNN:
    """RNN

    Yt+1 = A2f(A1Xt+C1Zt+b1)+b2
    Zt+1 = C2g(A1X+C1Zt+b1)+c2
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    @staticmethod
    def random(N=100, p=2):
        X = np.random.random(size=(N, p))
        return MLP(X, Y)


    def __call__(self, xs, h=1):
        Z = 0
        Xs = []
        for x in xs[:-h]:
            X, Z = _rnn(x, A1, b1, A2, b2, C1, c1, C2, c2, Z)
            Xs.append(X)
        E = LA.norm(_mlp(self.X, *x) - self.Y) / LA.norm(self.Y)
        return -E

