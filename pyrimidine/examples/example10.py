#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import SimpleBinaryIndividual, AgePopulation, AgeIndividual, SGAPopulation

from pyrimidine.benchmarks.cluster import *

# generate a knapsack problem randomly
evaluate = KMeans.random(N=100)


class YourIndividual(SimpleBinaryIndividual):

    def _fitness(self):
        return - evaluate(self)


class YourPopulation(SGAPopulation):
    element_class = YourIndividual

class MyIndividual(AgeIndividual, YourIndividual):
    pass

class MyPopulation(AgePopulation):
    element_class = MyIndividual
    life_span = 70



pop = YourPopulation.random(size=100)
_pop = pop.clone(type_=MyPopulation)

stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'}

# pop.evolve(verbose=True)

data = pop.history(ngen=200, stat=stat)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'])


pop = MyPopulation(individuals=_pop)

data = pop.history(ngen=200, stat=stat)

ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'])
ax.legend(('Fitness', 'Best Fitness', 'My Fitness', 'My Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
