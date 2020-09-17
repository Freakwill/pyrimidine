#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA

class KMeans:
    """KMeans clustering Problem

    min J(c,mu)=sum_i ||xi-mu_ci||
    where xi in ci
    """
    def __init__(self, X, n_components=2):
        self.X = X
        self.n_components = n_components

    @staticmethod
    def random(N, p=2):
        X1 = np.random.normal(1, 1, (N, p))
        X2 = np.random.normal(2, 1, (N, p))
        X = np.vstack((X1, X2))
        return KMeans(X, n_components=2)


    def __call__(self, x):
        cs = set(x)
        xs = {c:[self.X[i] for i, k in enumerate(x) if k==c] for c in cs}
        # mus = {c:np.mean(x, axis=0) for c, x in xs.items()}
        J = np.mean([np.sum([LA.norm(xi - np.mean(x, axis=0)) for xi in x]) for c, x in xs.items()])
        return J

# from scipy.stats import norm

# class MixGaussian:
#     """Mix Gaussian clustering Problem
#     X ~ sum a_i p(x|mu_i, S_i)

#     max L(ci,{mu_i,S_i})= prod_k p(xk|ci, {mu_i, S_i}) = prod_k sum_i a_i p(xk|mu_i, S_i)
#     """
#     def __init__(self, X, n_components=2):
#         self.X = X
#         self.n_components = n_components

#     @staticmethod
#     def random(N, p=2):
#         X1 = norm.rvs(size=(N, p))
#         X2 = norm.rvs(loc=2, size=(N, p))
#         X = np.vstack((X1, X2))
#         return MixGaussian(X, n_components=2)


#     def __call__(self, t):
#         cs = set(t)
#         xs = {c:[self.X[i] for i, k in enumerate(x) if k==c] for c in cs}
#         # mus = {c:np.mean(x, axis=0) for c, x in xs.items()}
#         J = np.sum([np.prod([norm.pdf(xi, ti) for xi in x]) for c, x in xs.items()])
#         return J
