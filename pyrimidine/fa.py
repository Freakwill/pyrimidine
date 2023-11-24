#!/usr/bin/env python3

"""Firefly Algorithm
"""

from .base import PopulationMixin
from .chromosome import FloatChromosome
from .individual import PolyIndividual
from .utils import gauss, random

from scipy.spatial.distance import pdist, squareform
import numpy as np

class BaseFirefly(BaseIndividual):

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

    def move(self, velocity=None):
        if velocity is None:
            self.position += self.velocity
        else:
            self.position += velocity

    def decode(self):
        return self.best_position

    def random_move(self):
        raise NotImplementedError


def attractiveness(distance, gamma=1.0):
    return np.exp(-gamma * distance**2)


class StandardFireflies:
    
    element_class = BaseFirefly

    params = {
    "gamma": 1,
    "beta": 1,
    "alpha": 0.2
    }
    
    def transition(self, *args, **kwargs):

        distances = squareform(pdist([f.position for f in self]))
        att = attractiveness(distances)

        for i, fi in enumerate(self[:-1]):
            for j, fj in enumerate(self[i+1:]):
                if fj.fitness > fi.fitness:
                    fi.move(self.alpha * att[i, j] * (fj.position - fi.position))

        for f in self:
            f.random_move()
