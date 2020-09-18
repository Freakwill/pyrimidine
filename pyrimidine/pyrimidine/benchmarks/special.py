#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def rosenbrock(n=5):
    def r(x):
        return sum((x[i+1]-x[i]**2)**2 * 100 + (x[i]-1)**2 for i in range(n-1))
    return r
