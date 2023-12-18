#!/usr/bin/env python3

import numpy as np

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
from pyrimidine.sma import *

X = np.linspace(-3, 3, 100)
y = np.arctan(X)

evaluate = Fitting(X, y)


class _Individual(FloatChromosome):

    def decode(self):
        return self[:10], self[10:20], self[20:30]

    def _fitness(self):
        return evaluate(*self.decode())


class _SlimyMaterial(_Individual, SlimyMaterial):

    def random_move(self):
        pass


class MyIndividual(_Individual):
    pass


MyPopulation = StandardPopulation[MyIndividual] // 30
YourPopulation = SlimeMould[_SlimyMaterial] // 30

pop1 = MyPopulation.random(size=30)
pop2 = pop1.copy(type_=YourPopulation)

pop1.ezolve(n_iter=50)
ind1 = pop1.best_individual
y1 = evaluate.fit(*ind1.decode())

pop2.ezolve(n_iter=50)
ind2 = pop2.best_individual
y2 = evaluate.fit(*ind2.decode())

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X, y, 'k', marker='o')
ax.plot(X, y1, 'r', X, y2, 'b')
ax.legend(('Original Function', f'With GA (Error: {-ind1.fitness})', f'With SMA (Error: {-ind2.fitness})'))
plt.show()
