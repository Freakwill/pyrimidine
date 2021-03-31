#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Particle Swarm Optimization

Developed by J. Kennedy and R. Eberhart[Kennedy and Eberhart 2001]

Each individual is represented position and velocity.
"""

from .base import BasePopulationModel
from .chromosome import FloatChromosome
from .individual import PolyIndividual
from .utils import gauss, random

import numpy as np

class BaseParticle(PolyIndividual):
    """A particle in PSO
    
    Extends:
        PolyIndividual
    
    Variables:
        default_size {number} -- one individual represented by 2 chromosomes: position and velocity
        memory {Particle} -- the best state of the particle moving in the solution space.

    Caution:
        implement the method-properties: position, velocity
    """

    element_class = FloatChromosome
    default_size = 2
    memory = None

    def backup(self):
        self.memory = self.clone(fitness=self.fitness)

    def init(self):
        self.backup()

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
        return self.memory.position

    @best_position.setter
    def best_position(self, x):
        self.memory.position = x

class Particle(BaseParticle):

    element_class = FloatChromosome
    default_size = 2
    memory = None

    @property
    def position(self):
        return self.chromosomes[0]

    @position.setter
    def position(self, x):
        self.chromosomes[0] = x
        self.fitness = None

    @property
    def velocity(self):
        return self.chromosomes[1]

    @velocity.setter
    def velocity(self, v):
        self.chromosomes[1] = v


class ParticleSwarm(BasePopulationModel):
    """Standard PSO
    
    Extends:
        BaseIterativeModel
    """
    
    element_class = Particle
    default_size = 20

    params = {'learning_factor': 2, 'acceleration_coefficient': 3, 'inertia':0.5, 'n_best_particles':0.1, 'max_velocity':None}

    def init(self):
        for particle in self.particles:
            particle.init()
        self.hall_of_fame = self.get_best_individuals(self.n_best_particles)
        

    # @property
    # def n_elements(self):
    #     return self.n_particles
    
    def update_fame(self):
        for particle in self.particles:
            if particle not in self.hall_of_fame:
                for k, fame in enumerate(self.hall_of_fame):
                    if particle.memory.fitness <= fame.memory.fitness:
                        break
                else:
                    self.hall_of_fame.pop(0)
                    self.hall_of_fame.insert(-1, particle)
                if k > 0:
                    self.hall_of_fame.pop(0)
                    self.hall_of_fame.insert(k-1, particle)
    
    def transit(self, *args, **kwargs):
        """
        Transitation of the states of particles
        """
        self.update_fame()
        self.move()

    def postprocess(self):
        for particle in self:
            # overwrite the memory of the particle if its current state is better its memory
            if particle.fitness > particle.memory.fitness:
                particle.backup()

    def move(self):
        """moving rule of particles

        Particles move according to the hall of fame and the best record
        """
        xi = random()
        eta = random()
        for particle in self:
            if particle in self.hall_of_fame:
                particle.velocity = (self.inertia * particle.velocity
             + self.learning_factor * xi * (particle.best_position-particle.position))
            else:
                for b in self.hall_of_fame:
                    if particle.fitness < b.fitness:
                        break
                particle.velocity = (self.inertia * particle.velocity
                 + self.learning_factor * xi * (particle.best_position-particle.position)
                 + self.acceleration_coefficient * eta * (b.best_position-particle.position))
            particle.position += particle.velocity

    @property
    def best_fitness(self):
        return np.max([_.memory.fitness for _ in self.hall_of_fame])
