#!/usr/bin/env python3

import numpy as np
from .benchmarks import Problem



class Knapsack(BaseProblem):
    """Knapsack Problem

    max sum_i ci xi
    s.t. sum_i wi xi <= W
    xi = 0 / 1 (choice variable)
    where ci is in c, wi is the coresponding weight of ci
    """
    def __init__(self, w, c, W=0.7, M=100):
        """
        
        Arguments:
            w {array} -- weight array of goods
            c {array} -- value array of goods
            W {number} -- upper bound (proportion) of total weight
        
        Keyword Arguments:
            M {number} -- penalty (default: {100})
        """

        if W < 1:
            W = np.sum(w) * W

        self.w = w
        self.c = c
        self.W = W
        self.M = M
        self.__sorted = None

    @staticmethod
    def random(n=50, W=0.7):
        w = np.random.randint(1, 21, n)
        c = np.random.randint(1, 21, n)
        return Knapsack(w, c, W=W)

    @staticmethod
    def example(W=0.7):
        w = [71, 34, 82, 23, 1, 88, 12, 57, 10, 68, 5, 33, 37, 69, 98, 24, 26, 83, 16, 26]
        c = [26, 59, 30, 19, 66, 85, 94, 8, 3, 44, 5, 1, 41, 82, 76, 1, 12, 81, 73, 32]
        return Knapsack(w, c, W=W)

    def argsort(self):
        return np.argsort(self.c / self.w)

    @property
    def sorted(self):
        if self.__sorted is None:
            self.__sorted =self.argsort()

        return self.__sorted

    def __call__(self, x):
        # x is a binary array
        c, w, W, M = self.c, self.w, self.W, self.M
        v = np.sum([ci for i, ci in zip(x, c) if i==1])
        w = np.sum([wi for i, wi in zip(x, w) if i==1])
        if  w <= W:
            return v
        else:
            return - 1/(1 + np.exp(-v)) * M


class MultiKnapsack(Problem):
    """Multi Choice Knapsack Problem

    max sum_ij cij xij
    s.t. sum_ij wij xij <= W
    {xij} is 
    where xij is the choice variable, that is sum_j xij=1
    """
    def __init__(self, ws, cs, W=0.7, M=100):
        """
        
        Arguments:
            ws {tuple of arrays} -- weight array of goods
            cs {tuple of arrays} -- value array of goods
            W {number} -- upper bound (proportion) of total weight
        
        Keyword Arguments:
            M {number} -- penalty (default: {100})
        """

        if W < 1:
            W = sum(map(np.sum, ws)) * W

        self.ws = ws
        self.cs = cs
        self.W = W
        self.M = M
        self.__sorted = None
        self.size = tuple(map(c, cs))

    @staticmethod
    def random(ns=(4,5,2,6,5,7,8,3), W=0.7):
        ws = tuple(np.random.randint(1, 21, n) for n in ns)
        cs = tuple(np.random.randint(1, 21, n) for n in ns)
        return MultiKnapsack(ws, cs, W=W)

    def argsort(self):
        c = min(self.cs)
        w = min(self.ws)
        return np.argsort(c / w)

    @property
    def sorted(self):
        if self.__sorted is None:
            self.__sorted =self.argsort()

        return self.__sorted

    def __call__(self, x):
        # x is an integer array
        c, w, W, M = self.c, self.w, self.W, self.M
        v = np.sum([ci[xi] for xi, ci in zip(x, c) if 0<=xi < self.size[i]])
        w = np.sum([wi[xi] for xi, wi in zip(x, w) if 0<=xi < self.size[i]])
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


from scipy.spatial.distance import pdist, squareform

class ShortestPath:
    def __init__(self, points):
        """TSP
        
        Arguments:
            points {array with shape of N * 2} -- points of the path
        """
        self.points = points
        self._dm = squareform(pdist(points))

    @staticmethod
    def random(N):
        return ShortestPath(np.random.random(size=(N, 2)))

    def __call__(self, x):
        return np.sum([self._dm[i,j] if i<j else self._dm[j, i] for i, j in zip(x[:-1], x[1:])])


class CurvePath(ShortestPath):
    def __init__(self, x, y):
        _points = np.column_stack([x,y])
        super().__init__(_points)


def _heart(t, a=1):
    x = np.cos(t)
    y = np.sin(t) + np.power(x**2, 1/3)
    return a * x, a * y

t = np.linspace(0, 2*np.pi, 50)
x1, y1 = _heart(t)
x2, y2 = _heart(t, a=1.3)
x = np.hstack((x1, x2))
y = np.hstack((y1, y2))
heart_path = CurvePath(x, y)


class MinSpanningTree:
    def __init__(self, nodes, edges=[]):
        self.nodes = nodes
        self.edges = edges

    def prufer_decode(self, x):
        P = x
        Q = set(self.nodes) - set(P)
        edges = []
        while P:
            i = min(Q)
            j = P[0]
            edges.append((i, j))
            Q.remove(i)
            P.pop(0)
            if j not in P:
                Q.add(j)
        edges.append(tuple(Q))
        return edges

class FacilityLayout(object):
    '''
    F: F
    D: D
    '''
    
    def __init__(self, F, D):
        self.F = F
        self.D = D
    
    @staticmethod
    def random(self, n):
        F = np.random.random(size=(n, n))
        D = np.random.random(size=(n, n))
        return FacilityLayout(F, D)

    def __call__(self, x):
        return np.dot(self.F.ravel(), np.array([self.D[xi, xj] for xj in x for xi in x]))
