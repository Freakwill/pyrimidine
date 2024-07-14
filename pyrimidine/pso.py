#!/usr/bin/env python3

"""Particle Swarm Optimization

Developed by J. Kennedy and R. Eberhart[Kennedy and Eberhart 2001]

Each individual consists of the position and the velocity. It also has the memory of the best solution

*Ref.*
J. Kennedy, R. Eberhart Particle swarm optimization Proc. IEEE Int. Conf. Neural Netw., 4 (1995), pp. 1942-1948
"""

from random import random
from operator import attrgetter

import numpy as np

from .base import BaseIndividual
from .chromosome import FloatChromosome
from .mixin import PopulationMixin
from .meta import MetaContainer
from .deco import basic_memory


@basic_memory
class BaseParticle(BaseIndividual):
    """A particle in PSO

    An individual represented by 2 chromosomes: position and velocity
    
    Variables:
        default_size {number} -- 2 by default
        memory {Particle} -- the best state of the particle moving in the solution space.

    Caution:
        Have to implement the properties in subclasses: position, velocity
    """

    element_class = FloatChromosome
    default_size = 2

    @property
    def position(self):
        raise NotImplementedError

    @position.setter
    def position(self, x):
        raise NotImplementedError

    @property
    def velocity(self):
        raise NotImplementedError

    @velocity.setter
    def velocity(self, v):
        raise NotImplementedError

    @property
    def best_position(self):
        # alias for the position of memory
        if self.memory.get('solution', None) is None:
            return self.position
        return self.memory['solution']

    def update_vilocity(self, fame=None, *args, **kwargs):
        raise NotImplementedError

    def move(self, velocity=None):
        if velocity is None:
            self.position += self.velocity
        else:
            self.position += velocity

    def decode(self):
        return self.best_position


class Particle(BaseParticle):
    """
    Standard Particle

    chromosomes = (position, velocity)
    """

    element_class = FloatChromosome
    default_size = 2

    params = {'learning_factor': 2,
        'acceleration_coefficient': 3,
        'inertia': 0.5
        }

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

    @property
    def direction(self):
        return self.best_position - self.position

    def update_vilocity(self, scale, inertia, learning_factor):      
        self.velocity = (inertia * self.velocity
                        + learning_factor * scale * self.direction)
    
    def update_vilocity_by_fame(self, fame, scale, scale_fame, inertia, learning_factor, acceleration_coefficient):
        self.velocity = (inertia * self.velocity
                        + learning_factor * scale * self.direction
                        + acceleration_coefficient * scale_fame * (fame.best_position-self.position))


class ParticleSwarm(PopulationMixin, metaclass=MetaContainer):
    """Standard PSO
    
    Extends:
        PopulationMixin
    """
    
    element_class = Particle
    default_size = 20

    params = {'learning_factor': 2, 'acceleration_coefficient': 3,
    'inertia':0.75, 'hof_size':0.2, 'max_velocity':None}

    alias = {
    'particles': 'elements',
    'n_particles': 'n_elements',
    'best_particle': 'best_element',
    'get_best_particle': 'get_best_element',
    'get_best_particles': 'get_best_elements'
    }

    def init(self):
        super().init()
        self.hall_of_fame = self.get_best_particles(self.hof_size, copy=True)
    
    def update_hall_of_fame(self):
        hof_size = len(self.hall_of_fame)
        for ind in self:
            for k in range(hof_size):
                if self.hall_of_fame[-k-1].fitness < ind.fitness:
                    self.hall_of_fame.insert(hof_size-k, ind.copy())
                    self.hall_of_fame.pop(0)
                    break

    @property
    def best_fitness(self):
        if self.hall_of_fame:
            return max(map(attrgetter('fitness'), self.hall_of_fame))
        else:
            return super().best_fitness

    def transition(self, *args, **kwargs):
        """
        Transitation of the states of particles
        """
        self.move()
        self.backup()
        self.update_hall_of_fame()

    def backup(self):
        # overwrite the memory of the particle if its current state is better its memory
        for particle in self:
            particle.backup()

    def move(self):
        """Move the particles

        Define the moving rule of particles, according to the hall of fame and the best record
        """

        scale = random()
        eta = random()
        scale_fame = random()
        for particle in self:
            for fame in self.hall_of_fame:
                if particle.fitness < fame.fitness:
                    particle.update_vilocity_by_fame(fame, scale, scale_fame, 
                        self.inertia, self.learning_factor, self.acceleration_coefficient)
                    particle.position = particle.position + particle.velocity
                    break
        for particle in self.hall_of_fame:
            particle.update_vilocity(scale, self.inertia, self.learning_factor)
            particle.position = particle.position + particle.velocity


class DiscreteParticleSwarm(ParticleSwarm):
    # PSO with discrete particles

    def move(self):
        """
        Move the discrete particles
        """

        v1, v2, v3 = (self.inertia,
            self.learning_factor,
            self.acceleration_coefficient)
        for particle in self:
            if particle in self.hall_of_fame:
                particle.velocity = _velocity1(particle, v1, v2)
            else:
                for fame in self.hall_of_fame:
                    if particle.fitness < fame.fitness:
                        break
                particle.velocity = _velocity2(particle, fame, v1, v2, v3)
            particle.move()
