#!/usr/bin/env python3


from pyrimidine.individual import MonoIndividual
from pyrimidine.population import HOFPopulation, BasePopulation
from pyrimidine.chromosome import FloatChromosome
from pyrimidine.local_search.simulated_annealing import *

from pyrimidine.benchmarks.optimization import *

from pyrimidine.benchmarks.special import *

n = 15
f = rosenbrock(n=n)

class MyIndividual(SimulatedAnnealing, MonoIndividual):
    element_class = FloatChromosome.set(default_size=n)

    def _fitness(self):
        return -f(self.chromosome)

    def get_neighbour(self):
        cpy = self.clone()
        cpy.mutate()
        # or cpy.chromosomes[0] = cpy.chromosome.random_neighbour()
        return cpy


ind = MyIndividual.random()

stat = {'Fitness':'_fitness'}
data = ind.evolve(stat=stat, n_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of Simulated Annealing')
plt.show()
