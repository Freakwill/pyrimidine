#!/usr/bin/env python3

import numpy as np

class Kantorovich(object):
    '''[Summary for Class Kantorovich]Kantorovich has 2 (principal) propteries
    a: a
    b: b [1]'''
    def __init__(self, a=0.5, b=1):
        self.a = a
        self.b = b
        self.x = np.linspace(a, b, 100)

    def __call__(self, p):
        # assert np.sum(p) == 1
        return np.dot(self.x, p) * np.dot(1/self.x, p)