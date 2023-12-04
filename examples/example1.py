#!/usr/bin/env python3

import numpy as np
from pyrimidine.chromosome import BinaryChromosome
from pyrimidine.population import HOFPopulation


t = np.random.randint(1, 5, 100)
n = np.random.randint(1, 4, 100)

import collections

def max_repeat(x):
    # maximum repetition
    c = collections.Counter(x)
    return np.max(list(c.values()))


def _evaluate(x):
    """
    select t_i, n_i from t, n resp.
    the sum of n_i is approx. to a given number,
    and t_i are repeated rarely
    """

    N = abs(np.sum([ni for ni, c in zip(n, x) if c==1]) - 30)
    T = max_repeat(ti for ti, c in zip(t, x) if c==1)
    return - (N + T /2)


class MyIndividual(BinaryChromosome // 10):

    def _fitness(self):
        return _evaluate(self.decode())


MyPopulation = HOFPopulation[MyIndividual] // 16

pop = MyPopulation.random()
stat = {'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=50, history=True, verbose=True)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.set_title('Demo for GA')
    plt.show()
