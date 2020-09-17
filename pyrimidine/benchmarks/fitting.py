#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.linalg as LA


_basis = [lambda x: np.ones(len(x)), lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4, 
np.sin, np.cos, np.tan, np.abs, lambda x:x>0, lambda x:x<0,
np.exp, lambda x: np.log(np.abs(x)+0.1)]

n_basis_ = len(_basis)

def relu(x):
    return x * (x>0)

class Fitting:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = LA.norm(y)

    def random(N, p):
        pass

    def fit(self, *params):
        return np.sum([c*relu(b * self.X - a) for a, b, c in zip(*params)], axis=0)


    def __call__(self, *params):
        yy = self.fit(*params)
        return - LA.norm(self.y-yy)/ self.n


_char = lambda x, y: x+2*y < 30 and x>=0 and y>=0

from math import cos, sin
def basis(a, t, d=1):
    c = cos(a)
    s = sin(a)
    return lambda xs, ys: np.array([[d * _char((c - 2*s) * (x - t[0]), (2*c+s) * (y - t[1]))
        for x in xs] for y in ys])

class Painting(Fitting):
    def __init__(self, image):
        self.size = image.size
        self.mode = image.mode
        from PIL_ext import image2output
        self.y = np.asarray(image, np.int8)

    def fit(self, *params):
        ks = np.arange(self.size[0])
        ls = np.arange(self.size[1])
        return np.sum([basis(theta, t, d)(ks, ls) for theta, t, d in zip(*params)], axis=0)

    def toimage(self, *params):
        yy = self.fit(*params)
        return Image.fromarray(yy.astype('uint8')).convert(self.mode)

    def __call__(self, *params):
        yy = self.fit(*params)
        return - LA.norm((self.y-yy) / 256)
