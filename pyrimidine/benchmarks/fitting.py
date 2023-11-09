#!/usr/bin/env python3


import numpy as np
import numpy.linalg as LA


_basis = [lambda x: np.ones(len(x)), lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4, 
np.sin, np.cos, np.tan, np.abs, lambda x:x>0, lambda x:x<0,
np.exp, lambda x: np.log(np.abs(x)+0.1)]

n_basis_ = len(_basis)


class Fitting:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = LA.norm(y)

    def random(N, p):
        pass

    def fit(self, *params):
        return np.sum([c*np.tanh(b * self.X - a) for a, b, c in zip(*params)], axis=0)


    def __call__(self, *params):
        yy = self.fit(*params)
        return - LA.norm(self.y-yy, 1) / self.n


class CurveFitting(Fitting):


    def fit(self, *params):
        return np.vstack((np.sum([c*relu(b * self.X - a) for a, b, c in zip(*params[:3])], axis=0), 
            np.sum([c*relu(b * self.X - a) for a, b, c in zip(*params[3:])], axis=0)))


_indicator = lambda x, y: np.and_(x+3*y < 2, x*y>=0)

from math import cos, sin
def basis(a, t):
    # basis in the form {di* indicator(Aix-ti)}
    c = cos(a)
    s = sin(a)
    def _f(x,y):
        y, x = np.meshgrid(y-t[0], x-t[1])
        return _indicator(c*x-s*y, s*x+c*y)
    return _f


from PIL import Image
class Painting(Fitting):
    def __init__(self, image, size=None, mode=None):

        self.size = size or image.size
        self.mode = mode or image.mode
        self.y = np.asarray(image.resize(self.size), np.float_)[:,:,0]
        self.n = 256

    def fit(self, *params):
        ks = np.arange(self.size[0])
        ls = np.arange(self.size[1])
        return np.sum([d * basis(theta, t)(ks, ls) for theta, t, d in zip(*params)], axis=0)

    def toimage(self, *params):
        yy = self.fit(*params)
        return Image.fromarray(yy.astype('uint8')).convert(self.mode)

