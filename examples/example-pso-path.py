#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.local_search import *
from pyrimidine.pso import Particle, ParticleSwarm
from pyrimidine.benchmarks.optimization import *


_evaluate = ShortestPath.random(30)


class _Chromosome(PermutationChromosome):
    default_size = 30

    def decode(self):
        return np.hstack((self, [self[0]]))

class _Individual(Particle, PolyIndividual):
    pass

def pminus(a, b):
    k = b.tolist().index(a[0])
    b = np.hstack((b[k:],b[:k]))
    swap = [k]
    i = 1
    while np.any(a[i:] != b[i:]):
        if a[i] != b[i]:
            l = b.tolist().index(a[i])
            swap.append(l)
            b[i], b[l] = b[l], b[i]
        i+=1
    return np.array(swap)

def vadd(a, v):
    k = v[0]
    a = np.hstack((a[k:],a[:k]))
    for i, l in enumerte(v[1:], 1):
        a[i], a[l] = a[l], a[i]
    return a


class _ParticleSwarm(ParticleSwarm, HOFPopulation):
    def move(self):
        """moving rule of particles

        Particles move according to the hall of fame and the best record
        """
        v1 = self.inertia
        v2 = self.learning_factor
        v3 = self.acceleration_coefficient
        for particle in self:
            if particle in self.hall_of_fame:
                particle.velocity = def_velocity1(particle, v1, v2)
            else:
                for fame in self.hall_of_fame:
                    if particle.fitness < fame.fitness:
                        break
                particle.velocity = def_velocity2(particle, fame, v1, v2, v3)
            particle.move()

_Particle = _Individual[_Chromosome].set_fitness(lambda obj: 1 / _evaluate(obj.decode()))

_Population = _ParticleSwarm[_Particle] // 80

pop = _Population.random()

from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
ax = fig.add_subplot(111)

points = _evaluate.points


def animate(i):
    pop.ezolve(n_iter=2)
    x = pop.best_individual.decode()
    ax.plot(points[x, 0], points[x, 1], 'k-o')
    ax.legend((f'Generation {i*2}({pop.best_fitness:.4})',))

camera = Camera(fig)
x = pop.best_individual.decode()
ax.plot(points[x, 0], points[x,1], 'k-o')
ax.legend((f'Generation 0({pop.best_fitness:.4})',))
for i in range(1, 801):
    animate(i)
    camera.snap()

animation = camera.animate()
animation.save('animation.mp4')
