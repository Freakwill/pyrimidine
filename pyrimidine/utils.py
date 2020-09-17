#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading

import operator
from random import random, randint, gauss

import numpy as np

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

# def gauss_pdf(x, mu=0, sigma=1):
#     return np.exp(-(x-mu)**2/sigma)

def randint2(lb=0, ub=9):
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
    return i, j

