#!/usr/bin/env python3


from pyrimidine import classicalIndividual
from pyrimidine.population import *

from pyrimidine.benchmarks.optimization import *

n_bags = 50
_evaluate = Knapsack.random(n_bags)

class MyIndividual(classicalIndividual(n_bags)):
    def _fitness(self):
        return _evaluate(self.chromosome)


class _Population1(StandardPopulation):
    element_class = MyIndividual
    default_size = n_bags

class _Population2(HOFPopulation):
    element_class = MyIndividual
    default_size = n_bags

pop1 = _Population1.random(size=n_bags)
# build population 2 with the same initial values to population 1, by `clone` method
pop2 = pop1.clone(type_=_Population2)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data1 = pop1.evolve(stat=stat, n_iter=300, history=True)
data2 = pop2.evolve(stat=stat, n_iter=300, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean 1', 'Best 1', 'Mean 2', 'Best 2'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
