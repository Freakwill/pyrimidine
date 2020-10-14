#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
import numpy as np

from pyrimidine.benchmarks.matrix import *

N=500
p=100
nmf = NMF.random(N, p)


class MyIndividual(MultiIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = ProbabilityChromosome

    def decode(self):
        A = np.column_stack((self[0], self[1], self[2]))
        B = np.column_stack((self[3], self[4], self[5]))
        return A, B, self[6]

    def _fitness(self):
        """
        select ni from n
        sum of ni ~ 10, while ti dose not repeat
        """
        return -nmf(*self.decode())


if __name__ == '__main__':
    SGAPopulation.element_class = MyIndividual

    pop = SGAPopulation.random(n_individuals=30, sizes=(N, N, N, p, p, p, 3))
    
    stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness'}
    data = pop.evolve(stat=stat, n_iter=100, history=True)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    plt.show()


    
