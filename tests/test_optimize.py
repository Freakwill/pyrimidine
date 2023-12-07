#!/usr/bin/env python3


from pyrimidine.optimize import ga_minimize

def test_ga_minimize():
    x = ga_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
    assert abs(x[0]) < 0.1 and x[1] < -0.9
