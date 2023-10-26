#!/usr/bin/env python

"""
Bat Algorithm
"""

from . import *

class Bat(PolyIndividual, MemoryIndividual):
    element_class = FloatChromosome
    default_size = 5
    phantom = None

    @property
    def position(self):
        return self.phantom.chromosomes[0]

    @position.setter
    def position(self, x):
        self.phantom.chromosomes[0] = x
        self.phantom.fitness = None

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
        self.fitness = None
