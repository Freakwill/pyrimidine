#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import operator
from random import random, randint, gauss, shuffle, choice
from math import exp

from scipy.spatial.distance import euclidean

import numpy as np
import numba as nb


class GOThread(threading.Thread):
    def __init__(self, target, *args, **kwargs):
        self.target = operator.methodcaller(target) if isinstance(target, str) else target
        super(GOThread, self).__init__(target=target, *args, **kwargs)


def parallel(func, individuals, *args, **kwargs):
    threads = [GOThread(target=func, args=(individual,)+args, kwargs=kwargs) for individual in individuals]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return [thread.result for thread in threads]

def binary_select(a, b, p=0.5):
    if random() < p:
        return a
    else:
        return b

from scipy.special import softmax
from scipy.stats import rv_discrete

def boltzmann_select(xs, fs, T=1):
    L = len(xs)
    ps = softmax(np.array(fs) /T)
    rv = rv_discrete(values=(np.arange(L), ps))
    k = rv.rvs()
    return xs[k]


def choice_with_prob(xs, ps, n=1):
    L = len(xs)
    ps /= np.sum(ps)
    X = np.arange(L)
    ks = []
    for _ in range(n):
        rv = rv_discrete(values=(np.arange(L), ps))
        k = rv.rvs()
        ks.append(X[k])
        X = np.delete(X, k)
        ps = np.delete(ps, k)
        ps /= np.sum(ps)
        L -= 1

    return [xs[k] for k in ks]


def choice_with_prob_replace(xs, ps, n=1):
    L = len(xs)
    ps /= np.sum(ps)
    rv = rv_discrete(values=(np.arange(L), ps))
    ks = rv.rvs(size=n)
    return [xs[k] for k in ks]


def choice_with_fitness(xs, fs=None, n=1, T=1):
    if fs is None:
        fs = [x.fitness for x in xs]
    ps = softmax(np.array(fs) /T)
    return choice_with_prob(xs, ps, n=1)


def choice_uniform(xs, n=1):
    L = len(xs)
    ks = np.random.choice(L, n)
    return [xs[k] for k in ks]


def randint2(lb=0, ub=9, ordered=False):
    """Select two different numbers in [lb, ub] randomly
    
    Formally i != j ~ U(lb, ub)
    Applied in GA operations.
    
    Keyword Arguments:
        lb {number} -- lower bound of interval (default: {0})
        ub {number} -- upper bound of interval (default: {9})
    
    Returns:
        two numbers
    """
    
    i = randint(lb, ub)
    d = ub - lb
    j = randint(i+1, d+i)
    if j > ub:
        j -= (d + 1)
    if ordered:
        if j < i:
            return j, i
    return i, j

@nb.vectorize()
def max0(x):
    return 0 if x<=0 else x

def max_lb(lb):
    @nb.vectorize()
    def m(x):
        return lb if x<=lb else x
    return m


@nb.vectorize()
def hl(x):
    return 0 if x<=0 else (1 if x>=1 else x)


def metropolis_rule(D, T, epsilon=0.000001):
    
    if D < 0:
        p = exp(D/max(T, epsilon))
        return random() < p
    else:
        return True
