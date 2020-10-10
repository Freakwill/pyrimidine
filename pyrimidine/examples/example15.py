#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

from digit_converter import *


from PIL import Image

evaluate = Painting(image=Image.open('/Users/William/Pictures/heart.jpg'), size=(100,100))


class _Gene(NaturalGene):
    lb, ub = 0, 100

class _Chromosome(VectorChromosome):
    element_class = _Gene

n_basis = 20

class MyIndividual(MixIndividual):
    element_class = UnitFloatChromosome, _Chromosome, BinaryChromosome

    def decode(self):
        c = self.chromosomes[2]
        c = c.reshape((n_basis, 3, 8))
        d = np.array([[colorConverter(c[i,j,:]) for j in range(c.shape[1])] for i in range(c.shape[0])])
        theta = self.chromosomes[0]
        t = self.chromosomes[1].reshape((n_basis, 2))
        return theta, t, d

    def _fitness(self):
        params = self.decode()
        return evaluate(*params)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=20, sizes=(n_basis, n_basis*2, 24*n_basis))


import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

def animate(i):
    pop.evolve(n_iter=2, verbose=False)
    params = pop.best_individual.decode()
    print(pop.best_fitness)
    im = evaluate.toimage(*params)
    plt.imshow(im)

camera = Camera(fig)
for i in range(50):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')
