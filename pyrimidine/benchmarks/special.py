#!/usr/bin/env python3

"""
Spcical functions for benchmark
"""

from math import sin, sqrt, pi
import numpy as np

def rosenbrock(x:np.ndarray):
    x = np.asarray(x)
    return np.sum( (100 * (x[1:]-x[:-1]**2)**2 + x[:-1]-1) ** 2 )


def schaffer(x:np.ndarray):
    r2 = np.sum(x**2)
    return (sin(sqrt(r2)) ** 2 - 0.5) / (1+ 0.001*r2**2)**2

def rastrigrin(x:np.ndarray):
    return np.sum(x**2 - 10 * np.cos(2*pi*x)) + 40


def griewangk(n=5):
    def f(x:np.ndarray):
        return np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
    return f


def michalewiez(n=5):
    def f(x:np.ndarray):
        return - sum(sin(x) * sin(np.arange(1, n+1) *x**2/pi)**20)/4000
    return f


def hansen(n=5):
    def f(x:np.ndarray):
        return np.sum(np.arange(1, n+1)*(np.cos(np.arange(n))*x+1)) * np.sum(np.arange(1, n+1)*(np.cos(np.arange(2, n+2))*x+1))
    return f


def alpine(x):
    return np.sum(np.abs((np.sin(x) + 0.1) * x))

