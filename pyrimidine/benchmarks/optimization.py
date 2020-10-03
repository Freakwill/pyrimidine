#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Knapsack:
    """Knapsack Problem

    max sum_i ci
    s.t. sum_i wi <= W
    where ci, wi selected from c, w
    """
    def __init__(self, w, c, W, M=100):
        """
        
        Arguments:
            w {array} -- weight array of goods
            c {array} -- value array of goods
            W {number} -- upper bound of total weight
        
        Keyword Arguments:
            M {number} -- penalty (default: {100})
        """

        self.w = w
        self.c = c
        self.W = W
        self.M = M

    @staticmethod
    def random(n=50, W=0.7):
        w = np.random.randint(1, 21, n)
        c = np.random.randint(1, 21, n)
        if W<1:
            W = np.sum(w) * W
        return Knapsack(w, c, W=W)


    def __call__(self, x):
        c, w, W, M = self.c, self.w, self.W, self.M
        v = np.sum([ci for i, ci in zip(x, c) if i==1])
        w = np.sum([wi for i, wi in zip(x, w) if i==1])
        if  w <= W:
            return v
        else:
            return - 1/(1 + np.exp(-v)) * M


class MLE:
    # max likelihood estimate
    def __init__(self, pdf, x):
        self.pdf = logpdf
        self.x = x

    @staticmethod
    def random(size=300):
        from scipy.stats import norm
        x = norm.rvs(size=size)
        return MLE(logpdf=norm.logpdf, x=x)

    def __call__(self, t):
        return np.sum([self.logpdf(xi, *t) for xi in self.x])


class MixMLE:
    # mix version of max likelihood estimate
    # x|k ~ pk
    def __init__(self, pdfs, x):
        self.pdfs = pdfs
        self.x = x

    @staticmethod
    def random(n_observants=300, n_components=2):
        from scipy.stats import norm
        x1 = norm.rvs(size=n_observants)
        x2 = norm.rvs(loc=2, size=n_observants)
        x = np.hstack((x1, x2))
        return MixMLE(pdfs=(norm.pdf,)*n_components, x=x)

    def logpdf(self, x, t, a):
        return np.log(np.dot([pdf(x, ti) for pdf, ti in zip(self.pdfs, t)], a))

    def __call__(self, t, a):
        # assert np.sum(a) == 1
        return np.sum([self.logpdf(xi, t, a) for xi in self.x])
