#!/usr/bin/env python3


from pyrimidine import *
from pyrimidine.benchmarks.optimization import *
from pyrimidine.deco import fitness_cache, basic_memory

# generate a knapsack problem randomly

n_bags = 100
evaluate = Knapsack.random(n_bags)


class _IndMixin:
    def _fitness(self):
        return evaluate(self.decode())


@basic_memory
class YourIndividual(_IndMixin, BinaryChromosome // n_bags):
    pass


@basic_memory
class MyIndividual(_IndMixin, QuantumChromosome // n_bags):

    def mutate(self):
        pass

    def cross(self, other):
        return self.__class__((self + other) /2)


class _Mixin:
    def init(self):
        for i in self: i.backup()
        super().init()

    def update_hall_of_fame(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """
        for i in self: i.backup()
        super().update_hall_of_fame(*args, **kwargs)


class MyPopulation(_Mixin, HOFPopulation):

    element_class = MyIndividual
    default_size = 20


class YourPopulation(_Mixin, HOFPopulation):

    element_class = YourIndividual
    default_size = 20


mypop = MyPopulation.random()
for i in mypop: i.measure()
yourpop = MyPopulation([YourIndividual(i.measure_result) for i in mypop])
mydata = mypop.evolve(n_iter=50, history=True)
yourdata = yourpop.evolve(n_iter=50, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
yourdata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
mydata[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean Fitness', 'Best Fitness', 'Mean Fitness(Quantum)', 'Best Fitness(Quantum)'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title(f'Demo of (Quantum)GA: {n_bags}-Knapsack Problem')
plt.show()
