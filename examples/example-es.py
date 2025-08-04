#!/usr/bin/env python3


from pyrimidine import FloatChromosome, BasePopulation
from pyrimidine.es import *
from pyrimidine.benchmarks.special import rosenbrock

import numpy as np
np.random.seed(6575)

n = 15
f = lambda x: -rosenbrock(x)

MyPopulation = EvolutionStrategy[FloatChromosome // n].set_fitness(f)


ind = MyPopulation.random()
stat = {'Mean Fitness':'mean_fitness', 'Best Fitness': 'max_fitness'}
data = ind.evolve(max_iter=100, stat=stat, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of Evolution Strategy')
plt.show()
