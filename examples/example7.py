#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
import numpy as np

from digit_converter import *

c = IntegerConverter()

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)


class MyIndividual(MultiIndividual[_Chromosome]):
    """base class of individual

    You should implement the methods, cross, mute
    """

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

    pop = SGAPopulation.random(n_individuals=30, size=15)
    data = pop.evolve(n_iter=100, stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}, history=True)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    plt.show()
