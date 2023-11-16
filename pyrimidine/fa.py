#!/usr/bin/env python3

"""Firefly Algorithm
"""

from .base import PopulationMixin
from .chromosome import FloatChromosome
from .individual import PolyIndividual
from .utils import gauss, random

import numpy as np

class Firefly(BaseIndividual):

    element_class = FloatChromosome
    default_size = 3

    params = {'gamma': 1,
    'alpha': 1
    }

    def backup(self):
        if self.memory is None:
            self.memory = self.clone(fitness=None)
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
        # alias for the position of memory
        return self.memory.position

    def update_vilocity(self, fame=None, *args, **kwargs):
        raise NotImplementedError

    def move(self):
        self.position += self.velocity

    def decode(self):
        return self.best_position
