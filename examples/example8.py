#!/usr/bin/env python3


from pyrimidine import *
from pyrimidine.local_search import *

from pyrimidine.benchmarks.optimization import *

_evaluate = heart_path


class _Chromosome(PermutationChromosome):
    default_size = 100

    def decode(self):
        return np.hstack((self, [self[0]]))

_Individual = MonoIndividual[_Chromosome].set_fitness(lambda i: - _evaluate(i.decode()))

_Population = HOFPopulation[_Individual] // 100

pop = _Population.random()
pop.init()

from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

points = _evaluate.points

def animate(i, step=5, start=0):
    pop.ezolve(n_iter=step)
    x = pop.best_individual.decode()
    ax.plot(points[x, 0], points[x, 1], 'r-o')
    ax.legend((f'Generation {start+i*step}({-pop.best_fitness:.4})',))

camera = Camera(fig)
x = pop.best_individual.decode()
ax.set_title('GA (with hall of fame) for TSP')
ax.plot(points[x, 0], points[x,1], 'r-o')
ax.legend((f'Generation 0({-pop.best_fitness:.4})',))
for i in range(1, 51):
    animate(i, step=2)
    camera.snap()
for i in range(1, 101):
    animate(i, start=50*2)
    camera.snap()
for i in range(1, 101):
    animate(i, step=10, start=50*2+100*5)
    camera.snap()
for i in range(1, 101):
    animate(i, step=20, start=50*2+100*5+100*10)
    camera.snap()

animation = camera.animate()
animation.save('anim-ga-tsp.mp4')
