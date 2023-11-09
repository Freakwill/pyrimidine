#!/usr/bin/env python3

from pyrimidine import *
import numpy as np


t = np.random.randint(1, 5, 100)
n = np.random.randint(1, 4, 100)

import collections
def max_repeat(x):
    # maximum repetition
    c = collections.Counter(x)
    return np.max([b for a, b in c.items()])


def _evaluate(x):
    """
    select ti, ni from t, n resp.
    the sum of ni ~ 10, while ti is repeated rarely
    """
    N = abs(np.sum([ni for ni, c in zip(n, x) if c==1])-30)
    T = max_repeat(ti for ti, c in zip(t, x) if c==1)
    return - (N + T /2)

class MyIndividual(BinaryChromosome.set(default_size=50)):

    def _fitness(self):
        return _evaluate(self.decode())

MyPopulation = HOFPopulation[MyIndividual] // 8

pop = MyPopulation.random()
stat = {'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=50, history=True)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    plt.show()
