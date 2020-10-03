#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BasePopulation
from .chromosome import FloatChromosome
from .individual import MultiIndividual
from .utils import gauss, random

import numpy as np

class Particle(MultiIndividual):
    # `phantom`: Particle -- the current state of the particle moving in the solution space.

    element_class = FloatChromosome
    default_size = 2

    def backup(self):
        self.chromosomes[0] = self.position
        self.fitness = self.phantom.fitness

    def init(self):
        self.phantom = self.clone()
        self.phantom.fitness = self.fitness

    @property
    def position(self):
        return self.phantom.chromosomes[0]

    @position.setter
    def position(self, x):
        self.phantom.chromosomes[0] = x

    @property
    def velocity(self):
        return self.phantom.chromosomes[1]

    @velocity.setter
    def velocity(self, v):
        self.phantom.chromosomes[1] = v

    @property
    def best_position(self):
        return self.chromosomes[0]

    @best_position.setter
    def best_position(self, x):
        self.chromosomes[0] = x


class ParticleSwarm(BasePopulation):
    """Standard Genetic Algo II.
    
    Extends:
        BasePopulation
    """
    element_class = Particle

    default_size = 20

    params = {'learning_factor': 2, 'acceleration_coefficient': 3, 'inertia':0.5, 'n_best_particles':0.1, 'max_velocity':None}

    def init(self):
        self.best_particles = self.get_best_individuals(self.n_best_particles)
        for particle in self.particles:
            particle.init()
    
    def transitate(self, k=None, *args, **kwargs):
        """
        Transitation of the states of population by SGA
        """
        for particle in self:
            if particle.phantom.fitness > particle.fitness:
                particle.backup()
        for particle in self:
            if particle not in self.best_particles:
                for k, b in enumerate(self.best_particles):
                    if particle.fitness <= b.fitness:
                        break
                if k > 0:
                    self.best_particles.pop(k)
                    self.best_particles.insert(k, particle)
        self.move()

    def move(self):
        xi = random()
        eta = random()
        for particle in self:
            if particle in self.best_particles:
                particle.velocity = (self.inertia * particle.velocity
             + self.learning_factor * xi * (particle.best_position-particle.position))
            else:
                for b in self.best_particles:
                    if particle.fitness < b.fitness:
                        break
                particle.velocity = (self.inertia * particle.velocity
                 + self.learning_factor * xi * (particle.best_position-particle.position)
                 + self.acceleration_coefficient * eta * (b.best_position-particle.position))
            particle.position += particle.velocity
            particle.phantom.fitness = None
