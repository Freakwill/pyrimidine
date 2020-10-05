#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

X = np.linspace(-3, 3, 10000)
y = np.arctan(1/ (np.abs(X)+0.1))

evaluate = Fitting(X, y)

class Gene1(FloatGene):
    lb, ub = -1, 1

class Chromosome1(FloatChromosome):
    element_class = Gene1

class Gene2(FloatGene):
    lb, ub = -3, 3

class Chromosome2(FloatChromosome):
    element_class = Gene2

class MyIndividual(MixIndividual):
    element_class = Chromosome1, Chromosome2, Chromosome2


    def _fitness(self):
        return evaluate(*self.chromosomes)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=20, sizes=(8, 8, 8))


import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

def animate(i):
    pop.evolve(n_iter=2, verbose=False)
    params = pop.best_individual.chromosomes
    yy = evaluate.fit(*params)
    ax.plot(X, y, '.r', X, yy, 'b')
    ax.legend(('Original Function', f'Approximating Function(Gen{i})'))

camera = Camera(fig)
for i in range(50):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')
