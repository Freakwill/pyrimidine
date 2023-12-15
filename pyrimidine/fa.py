#!/usr/bin/env python3

"""Firefly Algorithm
"""

from scipy.spatial.distance import pdist, squareform
import numpy as np

from .base import PopulationMixin
from .chromosome import FloatChromosome
from .individual import PolyIndividual
from .utils import gauss, random

from .pso import BaseParticle


@basic_memory
class BaseFirefly(BaseParticle):

    params = {
        'gamma': 1,
        'alpha': 1
    }

    def update_vilocity(self, fame=None, *args, **kwargs):
        raise NotImplementedError

    def random_move(self):
        raise NotImplementedError


def attractiveness(distance, gamma=1.0):
    return np.exp(-gamma * distance**2)


class StandardFireflies:
    """Starndard Firefly Algorithm
    """

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

        for f in self:
            f.backup()
