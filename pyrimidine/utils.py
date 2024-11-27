#!/usr/bin/env python3

"""
Helper functions
"""

from random import random, randint
from math import exp
from copy import deepcopy

from toolz import unique
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from scipy.stats import rv_discrete


def copy(obj, *args, **kwargs):
    if hasattr(obj, 'copy'):
        return obj.copy(*args, **kwargs)
    else:
        return deepcopy(obj)


def boltzmann_select(xs, fs, T=1):
    ps = softmax(np.array(fs) /T)
    rv = rv_discrete(values=(np.arange(len(xs)), ps))
    k = rv.rvs()
    return xs[k]


def choice(xs, *args, **kwargs):
    """Choose xi from xs with a certain probability
    
    Args:
        xs (List): a list of objects
    
    Returns:
        List: the sampling result
    """

    ks = np.random.choice(np.arange(len(xs)), *args, **kwargs)
    return [xs[k] for k in ks]


def choice_uniform(xs, *args, **kwargs):
    """Choose xi from xs with the unifrom probability
    
    Args:
        xs (List): a list of objects
    
    Returns:
        List: the sampling result
    """

    ks = np.random.randint(0, len(xs), *args, **kwargs)
    return [xs[k] for k in ks]


def choice_with_fitness(xs, fs=None, n=1, T=1):
    if fs is None:
        fs = [x.fitness for x in xs]
    ps = softmax(np.asarray(fs) / T)
    return choice(xs, p=ps, size=n)


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


def max0(x):
    return np.maximum(x, 0)


def hl(x):
    return np.clip(x, 0, 1) 


def metropolis_rule(D, T, epsilon=0.000001):
    """
    Metropolis rule
    
    Args:
        D (float): number representing the change of the value
        T (float): A number representing temperature
        epsilon (float, optional): The l.b. of T
    
    Returns:
        bool: change the state or not
    """
    if D < 0:
        p = exp(D / max(T, epsilon))
        return random() < p
    else:
        return True


# def proportion(n, N):
#     if n is None:
#         n = N
#     elif 0 < n < 1:
#         n = int(N * n)
#     return n


def pattern(chromosomes):
    """Get the pattern of the chromosomes
    
    Args:
        chromosomes (TYPE): A set of binary chromosomes
    
    Returns:
        str: the pattern of the chromosomes

    Example:
        >> pattern([[0,1,0], [1,0,0]])
        >> # Output "**0"
    """

    return ''.join([str(a[0]) if all(ai==a[0] for ai in a) else '*' for a in zip(*chromosomes)])


def rotations(x, y):
    """The rotations transforming x to y
    
    Args:
        x, y (array-like): a permutation of objects
    
    Returns:
        list of tuple: each tuple represent a rotation (based on indexes)

    Example:
        >> rotation([5,2,3,1,4], [2,5,3,4,1])
        >> [(0, 1), (3, 4)]
    """
    
    l = []
    for i, xi in enumerate(x):
        yi = y[i]
        if xi != yi:
            for j, t in enumerate(l):
                if x[t[-1]] == xi:
                    if x[t[0]] == yi:
                        break
                    else:
                        l[j] = (*t, x.index(yi))
                        break
                else:
                    if x[t[0]] == yi:
                        l[j] = (j, *t)
                        break
            else:
                l.append((i, x.index(yi)))
    return l


def rotate(x, rotations):
    """Permutate x by the list of rotations `rotation`
    
    Args:
        x (array-like): a permutation
        rotation (list of tuples): a list of rotations
    
    Returns:
        array-like:
    """
    
    if isinstance(rotations, tuple):
        r = rotations
        t = x[r[0]]
        for a, b in zip(r[:-1],r[1:]):
            x[a] = x[b]
        x[r[-1]] = t
    else:
        for r in rotations:
            x = rotate(x, r)
    return x


def prufer_decode(x, nodes=None):
    """Prufer code to tree
    
    Args:
        x (TYPE): Prufer code
        nodes (None, optional): all nodes of the tree
    
    Returns:
        list of pair, representing a tree

    Example:
        >> x = [2,5,6,8,2,5]
        >> print(prufer_decode(x))
        [(1, 2), (3, 5), (4, 6), (6, 8), (7, 2), (2, 5), (5, 8)]
    """
    if nodes is None:
        nodes = np.arange(1, len(x)+3)

    S = set(nodes) - set(x)
    edges = []
    while x:
        i = min(S)
        j = x[0]
        edges.append((i, j))
        S.remove(i)
        x.pop(0)
        if j not in x:
            S.add(j)
        print(x,S)
    edges.append(tuple(S))
    return edges
