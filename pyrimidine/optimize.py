#!/usr/bin/env python3


import numpy as np
import digit_converter

from .individual import BinaryIndividual
from .population import DualPopulation


def _decode(c, a, b):
    return IntervalConverter(a, b)(c)

def ga_min(fun, *xlim):
    # GA for minimizing fun defined on xlim

    class _Individual(BinaryIndividual):
        default_size = len(xlim)

        def _fitness(self):
            return - func(self.decode())

        def dual(self):
            return MyIndividual([c.dual() for c in self.chromosomes])

        def decode(self):
            return np.array([_decode(c, a, b) for c, (a, b) in zip(self.chromosomes, xlim)])

    class _Population(DualPopulation):
        element_class = _Individual
        default_size = 20

    pop = _Population.random()
    pop.ezolve()

    return pop.best_individual.decode()

