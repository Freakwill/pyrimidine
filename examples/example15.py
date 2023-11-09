#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.benchmarks.fitting import *
import numpy as np

from digit_converter import colorConverter


from PIL import Image

evaluate = Painting(image=Image.open('logo.png'), size=(100,100))


class _Gene(NaturalGene):
    lb, ub = 0, 256

class _Chromosome(VectorChromosome):
    element_class = _Gene
    default_size = 10


n_basis = 20

class MyIndividual(MixedIndividual):

    element_class = (UnitFloatChromosome // n_basis, _Chromosome // (n_basis*2), BinaryChromosome // (8*n_basis))

    def decode(self):
        c = self.chromosomes[2]
        c = c.reshape((n_basis, 8))
        d = np.array([colorConverter(c[i,:]) for i in range(c.shape[0])])
        theta = self.chromosomes[0]
        t = self.chromosomes[1].reshape((n_basis, 2))
        return theta, t, d

    def _fitness(self):
        params = self.decode()
        return evaluate(*params)


class MyPopulation(StandardPopulation):
    element_class = MyIndividual

pop = MyPopulation.random(n_individuals=20)


pop.ezolve(n_iter=50)
params = pop.best_individual.decode()
print(params)
im = evaluate.toimage(*params)
print(im)
im.show()

# from celluloid import Camera
# def animate(i):
#     pop.evolve(n_iter=2, verbose=False)
#     params = pop.best_individual.decode()
#     print(pop.best_fitness)
#     im = evaluate.toimage(*params)
#     plt.imshow(im)

# camera = Camera(fig)
# for i in range(50):
#     animate(i)
#     camera.snap()
# animation = camera.animate()
# animation.save('animation.mp4')
