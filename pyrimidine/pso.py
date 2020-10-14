#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseIterativeModel
from .chromosome import FloatChromosome
from .individual import PolyIndividual
from .utils import gauss, random

import numpy as np


class Particle(PolyIndividual):
    """A particle in PSO
    
    Extends:
        PolyIndividual
    
    Variables:
        default_size {number} -- one individual represented by 2 chromosomes: position and velocity
        phantom {Particle} -- the current state of the particle moving in the solution space.
    """

    element_class = FloatChromosome
    default_size = 2
    phantom = None

    def backup(self):
        self.chromosomes[0] = self.position
        self.fitness = self.phantom.fitness

    def init(self):
        self.phantom = self.clone(fitness=self.fitness)

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


class ParticleSwarm(BaseIterativeModel):
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

    # @property
    # def n_elements(self):
    #     return self.n_particles
    
    
    def transit(self, *args, **kwargs):
        """
        Transitation of the states of particles
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
        # moving rule of particles
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
