#!/usr/bin/env python3


from pyrimidine import *
from pyrimidine.local_search import *
from pyrimidine.utils import randint2


from pyrimidine.benchmarks.optimization import *


_evaluate = ShortestPath.random(30)


class _Chromosome(PermutationChromosome):
    default_size = 30

    def decode(self):
        return np.hstack((self, [self[0]]))

    def to_points(self):
        x = self.decode()
        return points[x, 0], points[x, 1]


_Individual = MonoIndividual[_Chromosome].set_fitness(lambda obj: - _evaluate(obj.decode()))


class SAIndividual(SimulatedAnnealing, _Individual):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        cpy.chromosome.mutate()
        return cpy

sa = SAIndividual.random(size=30)

from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

points = _evaluate.points

def animate(i):
    sa.evolve(n_iter=5, verbose=False)
    ax.plot(*sa.chromosome.to_points(), 'k-o')
    ax.plot(*sa.phantom.chromosome.to_points(), 'b--o')
    ax.legend((f'Best Solution({sa.fitness:.4})', f'Generation {i*5}'))

camera = Camera(fig)
ax.plot(*sa.chromosome.to_points(), 'k-o')
ax.legend(('Generation 0',))
for i in range(1, 300):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation-sa.mp4')

