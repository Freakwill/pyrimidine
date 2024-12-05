#!/usr/bin/env python3


from pyrimidine.population import StandardPopulation
from pyrimidine.optimize import ga_minimize, de_minimize, ga_minimize_1d, Optimizer


def test_ga_minimize():
    x = ga_minimize(lambda x: x[0]**2+x[1], (-1,1), (-1,1))
    assert abs(x[0]) < 0.2 and x[1] < -0.8


def test_ga_minimize_1d():
    x = ga_minimize_1d(lambda x: x**2, (-1,1))
    assert abs(x) < 0.2


def test_de_minimize():
    x = de_minimize(lambda x: x[0]**2+x[1], (-1,1), (-1,1))
    assert np.sum(np.abs(x- [0, -1])) < 0.1


def test_optimizer():
    optimizer = Optimizer(StandardPopulation)
    optimizer(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
    assert True