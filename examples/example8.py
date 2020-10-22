#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *


from pyrimidine.benchmarks.optimization import *


_evaluate = ShortestPath.random(30)


class _Chromosome(PermutationChromosome):
    default_size = 30

    def decode(self):
        return np.hstack((self, [self[0]]))


_Individual = MonoIndividual[_Chromosome].set_fitness(lambda obj: 1 / _evaluate(obj.decode()))

_Population = SGAPopulation[_Individual] * 40

MySpecies = DualSpecies[_Population]

sp = MySpecies.random()

from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

points = _evaluate.points

def animate(i):
    sp.ezolve(n_iter=2)
    x = sp.best_individual.decode()
    ax.plot(points[x, 0], points[x, 1], 'k-o')
    ax.legend(('Best Solution', f'Generation {i*2}'))

camera = Camera(fig)
x = sp.best_individual.decode()
ax.plot(points[x, 0], points[x,1], 'k-o')
ax.legend(('Generation 0',))
for i in range(1, 801):
    animate(i)
    camera.snap()
animation = camera.animate()
animation.save('animation.mp4')

