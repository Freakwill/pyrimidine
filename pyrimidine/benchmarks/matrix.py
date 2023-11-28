#!/usr/bin/env python3


import numpy as np
import numpy.linalg as LA


class NMF:
    # M ~ A diag(C) B'

    def __init__(self, M):
        self.M = M

    @staticmethod
    def random(N=500, p=100):
        M = np.random.rand(N, p) * 10
        s = M.sum(axis=1)
        for k in range(N):
            M[k] /= s[k]
        return NMF(M=M)

    def __call__(self, A, B, C=None):
        """A: N * K
        C: K
        B: K * p
        """

        c = A.shape[1]
        if C is not None:
            for i in range(c):
                A[:,i] *= C[i]
        return - LA.norm(self.M - np.dot(A, B))


class SparseMF:
    # M ~ A B'

    def __init__(self, M, threshold=0.01):
        self.M = M
        self.threshold = threshold

    @staticmethod
    def random(N=500, p=100):
        M = np.random.rand(N, p) * 10
        s = M.sum(axis=1)
        for k in range(N):
            M[k] /= s[k]
        return NMF(M=M)

    def __call__(self, A, B, C=None):
        """A: N * K
        C: K
        B: K * p
        """
        
        c = A.shape[1]
        if C is not None:
            for i in range(c):
                A[:,i] *= C[i]
        T = B > threshold
        B *= T
        return - LA.norm(self.M - np.dot(A, B)) - np.sum(T)
