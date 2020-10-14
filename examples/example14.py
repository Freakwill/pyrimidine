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
    lb, ub = -4, 4

class Gene3(FloatGene):
    lb, ub = -2, 2

class Chromosome2(FloatChromosome):
    element_class = Gene2

class Chromosome3(FloatChromosome):
    element_class = Gene3

class MyIndividual(MixIndividual):
    element_class = Chromosome1, Chromosome2, Chromosome3, Chromosome1, Chromosome2, Chromosome3


    def _fitness(self):
        return evaluate(*self.chromosomes)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    default_size = 40

class MySpecies(DualSpecies):
    element_class = MyPopulation

    def mate(self):
        male_offspring = []
        female_offspring = []
        children = [male.cross(female) for male, female in zip(self.males, self.females) if random()<0.75]
        male_offspring.extend(children[::2])
        female_offspring.extend(children[1::2])
        for _ in range(2):
            shuffle(self.females)
            children = [male.cross(female) for male, female in zip(self.males, self.females) if random()<0.75]
            male_offspring.extend(children[::2])
            female_offspring.extend(children[1::2])

        self.populations[0].individuals += male_offspring
        self.populations[1].individuals += female_offspring

pop = MySpecies.random(sizes=(20, 20, 20, 20, 20, 20))


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
    ax.legend(('Original Curve', f'Approximation(Gen{i*2})'))

camera = Camera(fig)
for i in range(500):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')
