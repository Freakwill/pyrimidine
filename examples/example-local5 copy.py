#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from pyrimidine.utils import randint2


from pyrimidine.benchmarks.optimization import *


_evaluate = ShortestPath.random(20)


class _Chromosome(PermutationChromosome):
    default_size = 20

    def decode(self):
        return np.hstack((self, [self[0]]))


class _Individual(MonoIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = _Chromosome

    def _fitness(self):
        return 1 / _evaluate(self.decode())

    def decode(self):
        return self.chromosome.decode()


class SAIndividual(SimulatedAnnealing, _Individual):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        cpy.chromosome.mutate()
        return cpy

sa = SAIndividual.random(size=20)

from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

# sa_data = sa.get_history(n_iter=500)
# ax.plot(sa_data.index, sa_data['Fitness'])
# plt.show()

points = _evaluate.points

def animate(i):
    sa.evolve(n_iter=1, verbose=False)
    x = sa.decode()
    ax.plot(points[x, 0], points[x, 1], 'k-o')
    y = sa.phantom.decode()
    ax.plot(points[y, 0], points[y, 1], 'b--o')
    ax.legend(('Best Solution', f'Generation {i*2}'))

camera = Camera(fig)
x = sa.decode()
ax.plot(points[x, 0], points[x,1], 'k-o')
ax.legend(('Generation 0',))
for i in range(1, 1501):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')

