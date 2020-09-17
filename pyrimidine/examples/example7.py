#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from beagle import *
import numpy as np

from digit_converter import *

c =IntegerConverter()

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)


class MyIndividual(MultiIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = _Chromosome
    default_size = 3

    def decode(self):
        x = super(MyIndividual, self).decode()
        return 2*x[0]+1, x[1]+1, 2*x[2]+1

    def _fitness(self):
        """
        select ni from n
        sum of ni ~ 10, while ti dose not repeat
        """
        x = self.decode()
        return -abs(x[0]**3+x[1]**3-x[2]**3) + 10*3 * min(abs(x[1]/x[2]-1), abs(x[0]/x[2]-1))

    def __str__(self):
        x = self.decode()
        return f'{x[0]}**3+{x[1]}**3={x[2]}**3+{x[0]**3+x[1]**3-x[2]**3}'



if __name__ == '__main__':
    SGAPopulation.element_class = MyIndividual

    pop = SGAPopulation.random(n_individuals=30, n_chromosomes=3, size=15)
    d= pop.history(ngen=100, stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'})
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')
    plt.show()
