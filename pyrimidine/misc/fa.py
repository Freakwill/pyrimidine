#!/usr/bin/env python3

"""
Firefly Algorithm

*References*
Yang, X. S. (2010). "Firefly Algorithm: A New Approach for Optimization".
In Stochastic Algorithms: Foundations and Applications (pp. 169-178). Springer Berlin Heidelberg.
Yang, X. S. (2013). "Nature-Inspired Metaheuristic Algorithms". Luniver Press.
"""


import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..mixin import PopulationMixin
from ..chromosome import FloatChromosome

from ..pso import BaseParticle


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


class StandardFireflies(PopulationMixin):
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
