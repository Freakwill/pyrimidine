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


def _evaluate(x):
    """
    select ti, ni from t, n
    sum of ni ~ 10, while ti dose not repeat
    """
    T = abs(np.sum([ni for ni, c in zip(n, x) if c==1])-10)
    N = max_repeat(ti for ti, c in zip(t, x) if c==1)
    return - (T + N)

class MyIndividual(MonoBinaryIndividual):

    def _fitness(self):
        return _evaluate(self.decode())

class MyPopulation(StandardPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=20, size=20)
stat = {'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=100, history=True)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    plt.show()
