#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
import numpy as np


t = np.random.randint(1, 5, 20)
n = np.random.randint(1, 4, 20)

import collections
def max_repeat(x):
    # maximum of numbers of repeats
    c = collections.Counter(x)
    bm=np.argmax([b for a, b in c.items()])
    return list(c.keys())[bm]


class MyIndividual(MonoBinaryIndividual):

    def _fitness(self):
        """
        select ti, ni from t, n
        sum of ni ~ 10, while ti dose not repeat
        """
        x, y = self.evaluate()
        return - (x + y)

    def evaluate(self):
        return abs(np.sum([ni for ni, c in zip(n, self.chromosome) if c==1])-10), max_repeat(ti for ti, c in zip(t, self.chromosome) if c==1)

class MyPopulation(SGAPopulation):
    element_class = MyIndividual

if __name__ == '__main__':
    pop = MyPopulation.random(n_individuals=20, size=20)
    # pop.evolve()
    # print(pop.best_individual)
    stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
    data = pop.evolve(stat=stat, n_iter=100, history=True)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    plt.show()
