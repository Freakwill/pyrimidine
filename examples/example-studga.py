#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import MonoBinaryIndividual
from pyrimidine.studga import *

from pyrimidine.benchmarks.optimization import *

n = 50
_evaluate = Knapsack.random(n)


# Define individual
class MyIndividual(MonoBinaryIndividual):
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)
    
# MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))

# Define Population
class MyPopulation(StudPopulation):
    element_class = MyIndividual
    default_size = 40

class YourPopulation(HOFPopulation):
    element_class = MyIndividual
    default_size = 40

pop = MyPopulation.random(size=n)
pop2 = pop.clone(type_=YourPopulation)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness', 'Standard Deviation of Fitnesses': 'std_fitness'}
data = pop.evolve(stat=stat, n_iter=200, history=True)
stat={'Mean Fitness2':'fitness', 'Best Fitness2':'best_fitness', 'Standard Deviation of Fitnesses2': 'std_fitness'}
data2 = pop2.evolve(stat=stat, n_iter=200, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
data2[['Mean Fitness2', 'Best Fitness2']].plot(ax=ax)
ax.legend(loc='upper left')
data['Standard Deviation of Fitnesses'].plot(ax=ax2, style='r--')
data2['Standard Deviation of Fitnesses2'].plot(ax=ax2, style='k--')
ax2.legend(loc='lower right')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
