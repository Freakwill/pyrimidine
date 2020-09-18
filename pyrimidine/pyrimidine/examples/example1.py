#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
import numpy as np


t = np.random.randint(1, 5, 100)
n = np.random.randint(1, 4, 100)

import collections
def max_repeat(x):
    # maximum of numbers of repeats
    c = collections.Counter(x)
    bm=np.argmax([b for a, b in c.items()])
    return list(c.keys())[bm]


class MyIndividual(SimpleBinaryIndividual):

    def _fitness(self):
        """
        select ti, ni from t, n
        sum of ni ~ 10, while ti dose not repeat
        """
        x = self.evaluate()
        return - x[0] - x[1]

    def evaluate(self):
        return abs(np.sum([ni for ni, c in zip(n, self) if c==1])-10), max_repeat(ti for ti, c in zip(t, self) if c==1)

class MyPopulation(SGAPopulation):
    element_class = MyIndividual

if __name__ == '__main__':
    pop = MyPopulation.random(n_individuals=50, size=100)
    # pop.evolve()
    # print(pop.best_individual)
    d = pop.history(ngen=200, stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'})
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(d.index, d['Fitness'], d.index, d['Best Fitness'], '.-')
    plt.show()
