#!/usr/bin/env python3

from pyrimidine import *

from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random(n=100)


class YourIndividual(classicalIndividual(size=100)):

    def _fitness(self):
        return evaluate(self.chromosome)


class YourPopulation(HOFPopulation):
    element_class = YourIndividual
    default_size = 20

class MyIndividual(AgeIndividual, YourIndividual):
    pass

class MyPopulation(AgePopulation):
    element_class = MyIndividual
    life_span = 70
    default_size = 20


pop = YourPopulation.random()
cpy = pop.clone(type_=MyPopulation)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness'}

# pop.evolve(verbose=True)

data = pop.evolve(n_iter=200, stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)

pop = MyPopulation(cpy.individuals)
data = pop.evolve(n_iter=200, stat=stat, history=True)

data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Fitness', 'Best Fitness', 'My Fitness', 'My Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
