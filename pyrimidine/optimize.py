#!/usr/bin/env python3

"""
Helpers for GA
"""

import numpy as np

from .individual import BinaryIndividual
from .population import HOFPopulation


def _decode(c, a, b):
    """Decode a binary sequence to a real number in [a,b]
    
    Args:
        c (binary seqence): Description
        a (number): lower bound
        b (number): upper bound
    
    Returns:
        number: a number in [a, b], represented by binary sequence,
        that 00...0 corresponds a, and 11...1 corresponds b
    
    Raises:
        e: fail to import a 3rd-part lib
    """
    try:
        from digit_converter import IntervalConverter
    except ImportError as e:
        print('The default decoder requires the 3rd-part lib `digit_converter`')
        raise e
    return IntervalConverter(a, b)(c)


def ga_min(func, *xlim, decode=_decode):
    """
    GA(with hof) for minimizing fun defined on xlim

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
