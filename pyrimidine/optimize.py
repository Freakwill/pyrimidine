#!/usr/bin/env python3

import numpy as np
from digit_converter import IntervalConverter

from .individual import BinaryIndividual
from .population import HOFPopulation


def _decode(c, a, b):
    return IntervalConverter(a, b)(c)


def ga_min(func, *xlim):
    """
    GA for minimizing fun defined on xlim

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
            return np.array([_decode(c, a, b) for c, (a, b) in zip(self.chromosomes, xlim)])

    class _Population(HOFPopulation):
        element_class = _Individual
        default_size = 20

    pop = _Population.random()
    pop.ezolve()

    return pop.best_individual.decode()
