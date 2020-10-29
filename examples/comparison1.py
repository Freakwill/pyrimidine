#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle

from digit_converter import *
from itertools import product

c = unitIntervalConverter

# generate a knapsack problem randomly

n_bags = 100
_evaluate = Knapsack.random(n=n_bags)

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self.chromosomes[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def expect(self):
        return c(self.chromosomes[1])
    

class _Population(BasePopulation):
    element_class = _Individual
    default_size = 30


class MySpecies(DualSpecies):
    element_class = _Population
    params = {'mate_prob': 0.5}


    def match(self, male, female):
        return male.expect >= female.ranking and female.expect >= male.ranking

    def mate(self):  
        self.populations[0].rank(tied=True)
        self.populations[1].rank(tied=True)
        children = [male.cross(female) for male, female in product(self.males, self.females) if random()<self.mate_prob and self.match(male, female)]
        self.populations[0].add_individuals(children[::2])
        self.populations[1].add_individuals(children[1::2])

    # def mutate(self):
    #     super(MySpecies, self).mutate()
    #     if random() < 0.5:
    #         best0 = self.populations[0].get_best_individuals(2)
    #         best1 = self.populations[1].get_best_individuals(2)
    #         if best0[0].fitness > best1[1].fitness:
    #             self.populations[1].add_individuals([best0[0].clone(), best0[1].clone()])
    #         elif best1[0].fitness > best0[1].fitness:
    #             self.populations[0].add_individuals([best1[0].clone(), best1[1].clone()])
    #         else:
    #             self.populations[1].add_individuals([best0[1].clone()])
    #             self.populations[0].add_individuals([best1[1].clone()])


class MyPopulation(_Population):
    default_size = 60

    def mate(self):
        for _ in range(5):
            shuffle(self.individuals)
            super(MyPopulation, self).mate()


sp = MySpecies.random(sizes=(n_bags, 4))
pop = MyPopulation(individuals=sp.clone().individuals)

stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness', 'Mean Fitness': 'mean_fitness'}

n_iter = 400
data2, t2 = sp.perf(stat=stat, n_iter=n_iter, n_repeats=1)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
data3, t3 = pop.perf(stat=stat, n_iter=n_iter, n_repeats=1)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data2.index * t2 / n_iter, data2['Best Fitness'], 'r',
    data2.index * t2 / n_iter, data2['Mean Fitness'], 'r--', 
    data3.index * t3 / n_iter, data3['Best Fitness'], 'b',
    data3.index * t3 / n_iter, data3['Mean Fitness'], 'b--')
ax.legend(('My Fitness', 'My Mean Fitness', 'Traditional Fitness', 'Traditional Mean Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.figsave('cmp.png')
