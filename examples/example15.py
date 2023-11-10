#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *

import numpy as np

from digit_converter import colorConverter

from PIL import Image

image = Image.open('painting.jpg')
evaluate = Painting(image=image, size=(100,100))


class _Gene(NaturalGene):
    lb, ub = 0, 100

class _Chromosome(VectorChromosome):
    element_class = NaturalGene
    default_size = 10


n_basis = 20

class MyIndividual(MixedIndividual):

    element_class = UnitFloatChromosome // (n_basis * 3), _Chromosome // (n_basis*2), BinaryChromosome // (8*n_basis)

    def decode(self):
        c = self.chromosomes[2]
        c = c.reshape((n_basis, 8))
        d = np.asarray([colorConverter(c[i,:]) for i in range(c.shape[0])])
        a, b, c = self.chromosomes[0][:n_basis], self.chromosomes[0][n_basis:2*n_basis], self.chromosomes[0][2*n_basis:]
        t = self.chromosomes[1].reshape((n_basis, 2))
        return a, b, c, t, d

    def _fitness(self):
        params = self.decode()
        return evaluate(*params)


class MyPopulation(HOFPopulation):
    element_class = MyIndividual


pop = MyPopulation.random(n_individuals=25)

pop.ezolve(n_iter=300)
params = pop.best_individual.decode()
im = evaluate.toimage(*params)
print(im)
im.show()

