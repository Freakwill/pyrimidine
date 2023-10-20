#!/usr/bin/env python3


from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

X = np.linspace(-3, 3, 100)
y = np.arctan(X)

evaluate = Fitting(X, y)

class Gene1(FloatGene):
    lb, ub = -1, 1

class Chromosome1(FloatChromosome):
    element_class = Gene1

class Gene2(FloatGene):
    lb, ub = -3, 3

class Chromosome2(FloatChromosome):
    element_class = Gene2

class _Individual(MixIndividual):
    element_class = Chromosome1, Chromosome2, Chromosome2


    def _fitness(self):
        return evaluate(*self.chromosomes)


MyPopulation = HOFPopulation[_Individual] // 50
YourPopulation = StandardPopulation[_Individual] // 50

pop = MyPopulation.random(sizes=(10, 10, 10))
pop2 = pop.clone(type_=YourPopulation)

pop.ezolve(n_iter=300)
ind1 = pop.best_individual
y1 = evaluate.fit(*ind1.chromosomes)

pop2.ezolve(n_iter=300)
ind2 = pop2.best_individual
y2 = evaluate.fit(*ind2.chromosomes)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X, y, X, y1, X, y2)
ax.legend(('Original Function', f'With hall of fame (Error: {-ind1.fitness})', f'Without (Error: {-ind2.fitness})'))
plt.show()
