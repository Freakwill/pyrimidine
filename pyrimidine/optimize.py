#!/usr/bin/env python3

"""
Helpers for the optimization based on GA.

Users can use the following example to optimize a multivariate function by GA directly
without encoding the solutions to chromosomes or individuals.

Example:

    ```
    # min x1^2+x2, x1,x2 in (-1,1)
    ga_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
    ```
"""


import numpy as np

from .chromosome import BinaryChromosome
from .individual import makeIndividual, makeBinaryIndividual
from .population import StandardPopulation
from .de import DifferentialEvolution


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


def _make_individual(func, *xlim, size=8, decode=_decode):
    """Make an individual class
    
    Args:
        *xlim: intervals
        size (int, tuple of int): the length of the encoding of xi
    """

    ndim = len(xlim)

    if ndim == 1:
        class _Individual(BinaryChromosome // size):

            def decode(self):
                return decode(self, *xlim[0])
    else:
        class _Individual(makeIndividual(n_chromosomes=ndim, size=size)):
            default_size = ndim

            def decode(self):
                return np.asarray([decode(c, a, b) for c, (a, b) in zip(self, xlim)])

    return _Individual.set_fitness(lambda obj: - func(obj.decode()))


def ga_minimize(func, *xlim, decode=_decode, population_size=20, size=8, **kwargs):
    """
    GA(with hall of fame) for minimizing the function `func` defined on `xlim`

    Arguments:

        func {function}: objective function defined on R^n
        xlim {tuple of number pairs}: the intervals of xi
        decode {mapping}: transform a binary sequence to a real number
            ('0-1' sequence, lower_bound, upper_bound) mapsto xi
        population_size {int}: size of the population
        size {int or tuple of int}: the length of the encoding of xi

    Example:

        ga_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
    """

    _Individual = _make_individual(func, *xlim, size=size, decode=decode)
    _Population = StandardPopulation[_Individual] // population_size

    pop = _Population.random()
    pop.ezolve(**kwargs)

    return pop.solution


def de_minimize(func, *xlim, decode=_decode, population_size=20, size=8, **kwargs):
    """
    DE for minimizing the function `func` defined on `xlim`

    Arguments:

        func {function}: objective function defined on R^n
        xlim: the intervals of xi
        decode: transform a binary sequence to a real number
            ('0-1' sequence, lower_bound, upper_bound) mapsto xi
        population_size: size of the population
        size: the length of the encoding of xi

    Example:

        de_minimize(lambda x:x[0]**2+x[1], (-1,1), (-1,1))
    """

    _Individual = _make_individual(func, *xlim, size=size, decode=decode)
    _Population = DifferentialEvolution[_Individual] // population_size

    pop = _Population.random()
    pop.ezolve(**kwargs)
    return pop.solution


def ga_minimize_1d(func, xlim, decode=_decode, population_size=20, size=8, *args, **kwargs):
    """
    GA(with hall of fame) for minimizing 1D function `func` defined on the interval `xlim`

    Arguments:
        func: objective function defined on R
        xlim {pair of numbers}: the interval of x
        decode: transform a binary sequence to a real number
            ('0-1' sequence, lower_bound, upper_bound) mapsto xi
        population_size {int}: size of the population
        size {int}: the length of the encoding of x

    Example:
        ga_minimize_1d(lambda x:x**2, (-1,1))
    """

    _Chromosome = _make_individual(func, xlim, size=size, decode=decode)
    _Population = StandardPopulation[_Chromosome] // population_size

    pop = _Population.random()
    pop.ezolve(**kwargs)
    return pop.solution


class Optimizer:

    """Optimizer class for optimization problem
    
    Attributes:
        min_max (str): 'min' or 'max'
        Population: GA population or other evolutionary algorithms
    """
    
    def __init__(self, Population=None, min_max='min'):
        self.Population = Population
        self.min_max = min_max

    def __call__(self, func, *xlim, decode=_decode, population_size=20, size=8, **kwargs):

        _Individual = _make_individual(func, *xlim, size=size, decode=decode)
        _Population = self.Population[_Individual] // population_size

        if self.min_max == 'min':
            def _evaluate(obj):
                return - func(obj.decode())
        else:
            def _evaluate(obj):
                return func(obj.decode())
        self.Population.set_fitness(_evaluate)

        pop = self.Population.random()
        pop.ezolve(**kwargs)
        return pop.solution

