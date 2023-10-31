#!/usr/bin/env python3


from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

X = np.linspace(-3, 3, 100)
y = np.arctan(X)

evaluate = Fitting(X, y)



class _Individual:
    element_class = FloatChromosome, FloatChromosome

    def decode(self):
        return self.chromosomes[0][:10], self.chromosomes[0][10:20], self.chromosomes[0][20:30]


    def _fitness(self):
        return evaluate(*self.decode())

class _Bat(_Individual, Bat):
    pass

class MyIndividual(_Individual, MixedIndividual):
    pass


MyPopulation = HOFPopulation[MyIndividual] // 50
YourPopulation = Bats[_Bat] // 50

pop = MyPopulation.random(sizes=(30, 30))
pop2 = pop.clone(type_=YourPopulation)

pop.ezolve(n_iter=500)
ind1 = pop.best_individual
y1 = evaluate.fit(*ind1.decode())

pop2.ezolve(n_iter=500)
ind2 = pop2.best_individual
y2 = evaluate.fit(*ind2.decode())

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X, y, X, y1, X, y2)
ax.legend(('Original Function', f'With GA (Error: {-ind1.fitness})', f'With BA (Error: {-ind2.fitness})'))
plt.show()
