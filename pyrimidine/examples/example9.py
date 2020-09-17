#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from beagle import SimpleBinaryIndividual, AgePopulation, AgeIndividual, SGAPopulation

from beagle.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random(200, W=0.6)

class MyIndividual(AgeIndividual, SimpleBinaryIndividual):
    life_span=5
    def _fitness(self):
        return evaluate(self)


class MyPopulation(AgePopulation):
    element_class = MyIndividual


class YourPopulation(SGAPopulation):
    element_class = MyIndividual

pop = YourPopulation.random(size=20)

stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'}
data, _ = pop.perf(stat=stat)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'])


pop = MyPopulation.random(size=20)

stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'}
data, _ = pop.perf(stat=stat)

ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'])
ax.legend(('Fitness', 'Best Fitness', 'My Fitness', 'My Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
