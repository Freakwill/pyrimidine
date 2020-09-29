#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA


class NMF:
    # M ~ A diag(C) B'
    def __init__(self, M):
        self.M = M

    @staticmethod
    def random(N=500, p=100):
        M = np.random.rand(N, p)
        s = M.sum(axis=1)
        for k in range(N):
            M[k] /= s[k]
        return NMF(M=M)


    def __call__(self, A, B, C=None):
        """A: N * K
        C: K
        B: K*p
        """
        r, c = A.shape
        if C:
            for i in range(c):
                A[:,i] *= C[i]
        return -LA.norm(self.M-np.dot(A, B.T))
