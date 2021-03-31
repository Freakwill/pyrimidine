#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.studga import *

from pyrimidine.benchmarks.optimization import *

n = 60
_evaluate = Knapsack.random(n)


# Define individual

class YourIndividual(MonoBinaryIndividual):
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosomes[0])

class MyIndividual(MixedIndividual):
    element_class = BinaryChromosome, FloatChromosome
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosomes[0])
    
# MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))

# Define Population
class MyPopulation(HOFPopulation, ModifiedPopulation):
    element_class = MyIndividual
    default_size = 40

class YourPopulation(HOFPopulation):
    element_class = YourIndividual
    default_size = 40

pop = MyPopulation.random(sizes=(n, 2))
pop2 = pop.clone(type_=YourPopulation)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness', 'Standard Deviation of Fitnesses': 'std_fitness'}
data = pop.evolve(stat=stat, n_iter=300, history=True)
stat={'Mean Fitness2':'fitness', 'Best Fitness2':'best_fitness', 'Standard Deviation of Fitnesses2': 'std_fitness'}
data2 = pop2.evolve(stat=stat, n_iter=300, history=True)


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
