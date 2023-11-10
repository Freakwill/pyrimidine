#!/usr/bin/env python3


from pyrimidine import AgeIndividual, MonoIndividual, BinaryChromosome, StandardPopulation, AgePopulation
from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random(200, W=0.6)

class MyIndividual(AgeIndividual, MonoIndividual):
    element_class = BinaryChromosome.set(default_size=200)
    life_span=5
    def _fitness(self):
        return evaluate(self.chromosome)


class MyPopulation(AgePopulation):
    element_class = MyIndividual


class YourPopulation(StandardPopulation):
    element_class = MyIndividual

pop = YourPopulation.random(size=20)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data, _ = pop.perf(stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)

pop = MyPopulation.random(size=20)
data, _ = pop.perf(stat=stat, history=True)

data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean Fitness', 'Best Fitness', 'My Fitness', 'My Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Knapsack Problem solved by GA')
plt.show()
