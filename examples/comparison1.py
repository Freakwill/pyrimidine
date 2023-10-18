#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle

from digit_converter import *
from itertools import product

c = unitIntervalConverter

# generate a knapsack problem randomly

import pathlib
p = pathlib.Path('easy200.txt')
lines = p.read_text().split('\n')

n_bags = int(lines[0])
cw = [line.strip(' ').split(' ')[1:] for line in lines[1:-1]]
c = [int(c) for c, w in cw]
w = [int(w) for c, w in cw]

_evaluate = Knapsack(w, c, W=int(lines[-1]))

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self.chromosomes[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def expect(self):
        return c(self.chromosomes[1])
    

_Population = HOFPopulation[_Individual] // 50


class MySpecies(DualSpecies):
    element_class = _Population
    params = {'mate_prob': 0.2}

    def init(self):
        self.populations[0].init()
        self.populations[1].init()

    def transit(self, k=None, *args, **kwargs):
        super().transit(*args, **kwargs)
        self.populations[0].update_halloffame()
        self.populations[0].add_individuals([i.clone() for i in self.populations[0].halloffame])
        self.populations[1].update_halloffame()
        self.populations[1].add_individuals([i.clone() for i in self.populations[1].halloffame])


    def match(self, male, female):
        return male.expect >= female.ranking and female.expect >= male.ranking

    def mate(self):
        self.populations[0].rank(tied=True)
        self.populations[1].rank(tied=True)
        children = [male.cross(female) for male, female in product(self.males, self.females) if random()<self.mate_prob and self.match(male, female)]
        self.populations[0].add_individuals(children[::2])
        self.populations[1].add_individuals(children[1::2])


class MyPopulation(_Population):
    default_size = 100

    def mate(self, mate_prob=None):
        offspring=[]
        for _ in range(5):
            shuffle(self.individuals)
            offspring.extend([individual.cross(other) for individual, other in zip(self.individuals[::2], self.individuals[1::2])
            if random() < (mate_prob or self.mate_prob)])

        self.add_individuals(offspring)
        self.offspring = self.__class__(offspring)


sp = MySpecies.random(sizes=(n_bags, 3))
pop = MyPopulation(individuals=sp.clone().individuals)

stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness', 'Mean Fitness': 'mean_fitness'}

n_iter = 500
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
ax.set_xlabel('Time')
ax.set_ylabel('Fitness')
plt.show()
