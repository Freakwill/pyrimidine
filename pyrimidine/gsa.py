#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseFitnessModel
from .chromosome import FloatChromosome
from .individual import PolyIndividual
from .utils import euclidean, random, exp, metropolis_rule

import numpy as np


class Particle(PolyIndividual):
    """A particle in GSA
    
    Extends:
        PolyIndividual
    
    Variables:
        default_size {number} -- one individual represented by 2 chromosomes: position and velocity
        phantom {Particle} -- the current state of the particle moving in the solution space.
    """

    element_class = FloatChromosome
    default_size = 2
    accelerate = 0


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

    def move(self): 
        r = random()
        cpy = self.clone(fitness=None)
        cpy.velocity = r * cpy.velocity + cpy.accelerate
        cpy.position += cpy.velocity
        flag = metropolis_rule(D=cpy.fitness - self.fitness, T=abs(self.fitness))
        if flag:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness


class GravitySearch(BaseFitnessModel):
    """Standard GSA
    
    Extends:
        BaseFitnessModel
    """

    element_class = Particle
    default_size = 20

    params = {'gravity_coefficient': 100, 'attenuation_coefficient': 20}

    def compute_mass(self):
        worst_fitness = np.min([particle.fitness for particle in self])
        best_fitness = np.min([particle.fitness for particle in self])
        epsilon = 0.00001
        q = (np.array([particle.fitness for particle in self]) - worst_fitness + epsilon) / (best_fitness - worst_fitness + epsilon)
        return q / q.sum()


    def compute_accelerate(self):
        # compute force
        D = np.array([[pj.position - pi.position for pi in self] for pj in self])
        R = np.array([[euclidean(pi.position, pj.position) for pi in self] for pj in self])**3

        M = self.compute_mass()
        MM = np.tile(M, (len(self), 1)) / R
        for i, p in enumerate(self):
            MM[i, i]=0
        MM = self.gravity_coefficient * MM * np.random.random((len(self), len(self)))
        A = np.array([MM * D[:,:, k] for k in range(len(self.particles[0]))])
        A = A.sum(axis=1)

        # compute accelerate
        for i, particle in enumerate(self):
            particle.accelerate = A[:, i]

    def postprocess(self):
        self.gravity_coefficient *= exp(-self.attenuation_coefficient/self.n_iter)

    
    def transit(self, *args, **kwargs):
        """
        Transitation of the states of particles
        """
        self.compute_accelerate()
        self.move()

    def move(self):
        """Moving particles with Newton's mechanics
        """
        for particle in self:
            particle.move()

