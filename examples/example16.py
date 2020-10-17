#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle

from digit_converter import *

c = unitIntervalConverter

# generate a knapsack problem randomly

n_bags = 100
_evaluate = Knapsack.random(n=n_bags)

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def expect(self):
        return c(self[1])


class _Population(BasePopulation):
    element_class = _Individual
    default_size = 20

    # def select(self):
    #     self.individuals = self.get_best_individuals(20)

class MySpecies(DualSpecies):
    element_class = _Population

    def match(self, male, female):
        return male.expect <= male.ranking and female.expect <= female.ranking

    @property
    def expect(self):
        return np.mean([ind.expect for ind in self.males] + [ind.expect for ind in self.females])

    @property
    def best_expect(self):
        return self.best_individual.expect


class YourSpecies(DualSpecies):
    element_class = _Population

    def mate(self):
        male_offspring = []
        female_offspring = []
        for male, female in zip(self.males, self.females):
            if random()<0.75:
                child = male.cross(female)
                if random()<0.5:
                    male_offspring.append(child)
                else:
                    female_offspring.append(child)
        shuffle(self.females)
        for male, female in zip(self.males, self.females):
            if random()<0.75:
                child = male.cross(female)
                if random()<0.5:
                    male_offspring.append(child)
                else:
                    female_offspring.append(child)

        self.populations[0].individuals += male_offspring
        self.populations[1].individuals += female_offspring

class MyPopulation(_Population):
    default_size = 40

    # def mate(self):
    #     super(MyPopulation, self).mate()
    #     shuffle(self.individuals)
    #     super(MyPopulation, self).mate()


sp = MySpecies.random(sizes=(n_bags, 3))
sp2 = sp.clone(type_=YourSpecies)
pop = MyPopulation(individuals=sp.populations[0].clone().individuals+sp.populations[1].clone().individuals)

stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness', 'Mean Expect': 'expect', 'Best Expect': 'best_expect'}

data, t = sp.perf(stat=stat, n_iter=200, n_repeats=1)

data.to_csv('h.csv')



stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness'}
data2, t2 = sp2.perf(stat=stat, n_iter=200, n_repeats=1)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
data3, t3 = pop.perf(stat=stat, n_iter=200, n_repeats=1)


print(t, t2, t3)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data['Best Fitness'].plot(ax=ax)

data2['Best Fitness'].plot(ax=ax, style='--')
data3['Best Fitness'].plot(ax=ax, style='-.')
ax.legend(('My Fitness', 'Your Fitness', 'Traditional Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
