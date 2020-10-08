#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

X = np.linspace(-3.14, 3.14, 200)
y = np.vstack((2*np.sin(X)-np.sin(2*X), 2*np.cos(X)-np.cos(2*X)))

evaluate = CurveFitting(X, y)

class Gene1(FloatGene):
    lb, ub = -1, 1

class Chromosome1(FloatChromosome):
    element_class = Gene1

class Gene2(FloatGene):
    lb, ub = -5, 5

class Chromosome2(FloatChromosome):
    element_class = Gene2

class MyIndividual(MixIndividual):
    element_class = Chromosome1, Chromosome2, Chromosome2, Chromosome1, Chromosome2, Chromosome2


    def _fitness(self):
        return evaluate(*self.chromosomes)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=40, sizes=(16, 16, 16, 16, 16, 16))


from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

def animate(i):
    pop.evolve(n_iter=2, verbose=False)
    params = pop.best_individual.chromosomes
    yy = evaluate.fit(*params)
    ax.set_xlim((-5,5))
    ax.set_ylim((-5,5))
    ax.plot(y[0], y[1], '.r', yy[0], yy[1], 'b')
    ax.legend(('Original Curve', f'Approximation(Gen{i})'))

camera = Camera(fig)
for i in range(500):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')
