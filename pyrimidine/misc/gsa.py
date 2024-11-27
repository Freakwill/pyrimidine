#!/usr/bin/env python3

"""
The Gravity Searching Algorithm (GSA) is a metaheuristic optimization method that simulates the law of gravity in physics.
It was introduced as a nature-inspired algorithm for solving optimization problems, particularly in continuous domains.

*References*
1. Rashedi, E., Nezamabadi-Pour, H., & Saryazdi, S. (2009). "GSA: A Gravitational Search Algorithm". Information Sciences, 179(13), 2232-2248.
2. Rashedi, E., Nezamabadi-Pour, H., & Saryazdi, S. (2011). "A New Method for Solving Optimization Problems Using Gravitational Search Algorithm". International Journal of Computer Applications, 22(8), 1-6.
3. Niazi, M., Mirjalili, S., Mirjalili, S. M., & Yang, X. S. (2016). "Enhanced Gravity Search Algorithm". Swarm and Evolutionary Computation, 6(1), 10-21.
"""

from scipy.spatial.distance import pdist, squareform
from ..mixin import PopulationMixin
from ..chromosome import FloatChromosome
from ..pso import BaseParticle
from ..utils import euclidean, random, exp, metropolis_rule
from ..deco import side_effect

import numpy as np


class Particle(BaseParticle):
    """A particle in GSA
    
    Extends:
        PolyIndividual
    
    Variables:
        default_size {number} -- one individual represented by 2 chromosomes: position and velocity
        phantom {Particle} -- the current state of the particle moving in the solution space.
    """

    element_class = FloatChromosome
    default_size = 2

    params = {'accelerate': 0}

    @property
    def position(self):
        return self.chromosomes[0]

    @position.setter
    def position(self, x):
        self.chromosomes[0] = x
        self.after_setter()

    @property
    def velocity(self):
        return self.chromosomes[1]

    @velocity.setter
    def velocity(self, v):
        self.chromosomes[1] = v

    @side_effect
    def move(self):
        """Moving the particl with Newton's mechanics
        """
        r = random()
        cpy = self.copy(fitness=None)
        cpy.velocity = r * cpy.velocity + cpy.accelerate
        cpy.position = cpy.position + cpy.velocity
        flag = metropolis_rule(D=cpy.fitness - self.fitness, T=10)
        D = cpy.fitness - self.fitness
        if flag:
            self.chromosomes = cpy.chromosomes


class GravitySearch(PopulationMixin):
    """Standard GSA
    
    Extends:
        PopulationMixin
    """

    alias = {'particles': 'elements',
    'n_particles': 'n_elements'}

    element_class = Particle
    default_size = 20

    params = {'gravity_coefficient': 100, 'attenuation_coefficient': 10}

    def compute_mass(self):
        fitnesses = np.asarray([particle.fitness for particle in self])
        worst_fitness = np.min(fitnesses)
        best_fitness = np.max(fitnesses)
        epsilon = 0.0001
        m = (fitnesses - worst_fitness + epsilon) / (best_fitness - worst_fitness + epsilon)
        return m / m.sum()

    def compute_accelerate(self):
        # compute force
        D = np.array([[pj.position - pi.position for pi in self] for pj in self])
        R = squareform(pdist([p.position for p in self]))
        for i in range(self.n_particles):
            R[i, i]=1
        m = self.compute_mass()
        M = np.tile(m, (self.n_particles, 1))
        for i in range(self.n_particles):
            M[i, i]=0
        M /= R**3
        M *= self.gravity_coefficient * np.random.random((self.n_particles, self.n_particles))
        A = M[:,:,None] * D
        A = A.sum(axis=0)

        # set accelerate
        for i, particle in enumerate(self):
            particle.accelerate = A[i, :]

    def transition(self, k):
        """
        Transitation of the states of particles
        """
        self.compute_accelerate()
        self.move()
        self.gravity_coefficient = exp(-self.attenuation_coefficient*k / self.max_iter)

    def move(self):
        for particle in self:
            particle.move()

