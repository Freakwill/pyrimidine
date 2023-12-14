#!/usr/bin/env python3


from pyrimidine import FloatChromosome, BasePopulation
from pyrimidine.es import *
from pyrimidine.benchmarks.special import rosenbrock


n = 15
f = lambda x: -rosenbrock(x)

MyPopulation = EvolutionStrategy[FloatChromosome // n].set_fitness(f)


ind = MyPopulation.random()

data = ind.evolve(n_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of Evolution Strategy')
plt.show()
