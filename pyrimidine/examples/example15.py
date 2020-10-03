#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

from digit_converter import *


from PIL import Image

evaluate = Painting(image=Image.open('/Users/William/Pictures/taiji.jpg'))


class _Gene(NaturalGene):
    lb, ub = 0, 100

class _Chromosome(VectorChromosome):
    element_class = _Gene

class MyIndividual(MixIndividual):
    element_class = UnitFloatChromosome, _Chromosome, BinaryChromosome

    def decode(self):
        c = self.chromosomes[2]
        c = c.reshape((8, 3, 8))
        d = np.array([[colorConverter(c[i,j,:]) for j in range(c.shape[1])] for i in range(c.shape[0])])
        theta = self.chromosomes[0]
        t = self.chromosomes[1].reshape((8, 2))
        return theta, t, d

    def _fitness(self):
        params = self.decode()
        return evaluate(*params)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=5, sizes=(8, 8*2, 8*3*8))


import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

def animate(i):
    pop.evolve(n_iter=2, verbose=False)
    params = pop.best_individual.decode()
    im = evaluate.toimage(*params)
    plt.imshow(im)

camera = Camera(fig)
for i in range(50):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')
