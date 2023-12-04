#!/usr/bin/env python3


import numpy as np

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
from digit_converter import colorConverter

from PIL import Image

image = Image.open('taichi.jpeg')
evaluate = Painting(image=image, size=(100,100))


class _Gene(NaturalGene):
    lb, ub = 0, 90

class _Chromosome(VectorChromosome):

    element_class = _Gene
    default_size = 10


n_basis = 20

from pyrimidine.deco import fitness_cache

@fitness_cache
class MyIndividual(MixedIndividual):

    element_class = UnitFloatChromosome // (n_basis * 3), _Chromosome // (n_basis*2), BinaryChromosome // (8*n_basis)

    def decode(self):
        c = self.chromosomes[2]
        c = c.reshape((n_basis, 8))
        c = np.asarray([colorConverter(ci) for ci in c])
        a, d1, d2 = self.chromosomes[0][:n_basis], self.chromosomes[0][n_basis:2*n_basis], self.chromosomes[0][2*n_basis:]
        t = self.chromosomes[1].reshape((n_basis, 2))
        return a, d1, d2, t, c

    def _fitness(self):
        params = self.decode()
        return evaluate(*params)


MyPopulation = HOFPopulation[MyIndividual] // 50

pop = MyPopulation.random()

pop.ezolve(n_iter=250)
params = pop.solution
im = evaluate.toimage(*params)
im.show()

