#!/usr/bin/env python3

"""
Helpers for GA
"""

import numpy as np

from .individual import BinaryIndividual
from .population import HOFPopulation


def _decode(c, a, b):
    from digit_converter import IntervalConverter
    return IntervalConverter(a, b)(c)


def ga_min(func, *xlim, decode=_decode):
    """
    GA for minimizing fun defined on xlim

    Arguments:
        func: objective function defined on R
        xlim: the intervals of xi
        decode: transform a binary sequence to a real number
            ('0-1' sequence, lower_bound, upper_bound) |-> xi

    Example:
        ga_min(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
    """

    class _Individual(BinaryIndividual):
        default_size = len(xlim)

        def _fitness(self):
            return - func(self.decode())

        def dual(self):
            return MyIndividual([c.dual() for c in self.chromosomes])

        def decode(self):
            return np.array([decode(c, a, b) for c, (a, b) in zip(self.chromosomes, xlim)])

    class _Population(HOFPopulation):
        element_class = _Individual
        default_size = 20

    pop = _Population.random()
    pop.ezolve()

    return pop.best_individual.decode()
