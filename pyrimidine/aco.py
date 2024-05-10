#!/usr/bin/env python3

"""Ant Colony Optimization

*Ref*
Blum, Christian. "Ant colony optimization: Introduction and recent trends." Physics of Life reviews 2.4 (2005): 353-373.
"""


from types import MethodType
import numpy as np
from .base import FitnessMixin, PopulationMixin
from .meta import MetaContainer
from scipy.stats import rv_discrete

from random import random


class BaseAnt(FitnessMixin):

    initial_position = 0

    params = {"initial_position": None,
    "path": None,
    "n_steps": 1}

    def init(self):
        self.path = [0]

    def __init__(self, *args, **kwargs):
        if args:
            raise Exception('__init__ has no position arguments!')
        for k, v in kwargs.items():
            setattr(self, k, v)

    def move(self, colony=None, n_steps=1):
        colony = colony or self._colony
        i = self.path[-1]
        self.new_path = [i]
        for _ in range(n_steps):
            next_positions = [j for j in colony.positions if j not in self.path]
            if not next_positions: break
            elif len(next_positions) == 1:
                i = next_positions[0]
            else:
                ps = np.array([colony.move_proba[i, j] for j in colony.positions if j not in self.path])
                ps += 0.0001
                ps /= np.sum(ps)
                rv = rv_discrete(values=(next_positions, ps))
                i = rv.rvs(size=1)[0]
            self.new_path.append(i)
            self.path.append(i)
        if len(self.new_path) >= 2:
            self.release_pheromone(colony)

    def release_pheromone(self, colony):
        self.pheromone = colony.sedimentation / np.sum(colony.distances[i,j] for i, j in zip(self.new_path[:-1], self.new_path[1:]))

    def get_length(self, distances=None):
        distances = distances or self._colony.distances
        return np.sum(distances[i,j] for i, j in zip(self.path[:-1], self.path[1:]))

    def _fitness(self):
        n = len(self._colony.positions) - len(self.path)
        if n == 0:
            return 1 / self.get_length()
        else:
            rest_positions = [self.path[-1]] + [j for j in self._colony.positions if j not in self.path]
            M = np.mean([self._colony.distances[i,j] for i in rest_positions for j in rest_positions if i<j])
            return 1 / (self.get_length() + n*M)

    def reset(self):
        self.path = [0]
        

class BaseAntColony(PopulationMixin, metaclass=MetaContainer):

    element_class = BaseAnt
    params = {'sedimentation':100, 'volatilization':0.75, 'alpha':1, 'beta':5, 'n_steps':1}

    alias = {"ants": "elements", "worst_ant": "worst_element", "get_worst_individuals": "get_worst_elements"}

    @classmethod
    def from_positions(cls, n_ants=10, positions=None):
        from scipy.spatial.distance import pdist, squareform
        obj = cls([cls.element_class() for _ in range(n_ants)])
        obj.positions = positions
        obj.pheromone = np.zeros((len(positions),)*2)
        obj.distances = squareform(pdist(points))
        obj.observe()
        return obj

    @classmethod
    def from_distances(cls, n_ants=10, distances=None):
        obj = cls([cls.element_class() for _ in range(n_ants)])
        obj.positions = np.arange(distances.shape[0])
        obj.distances = distances
        obj.pheromone = np.zeros(distances.shape)
        obj.observe()
        return obj

    def observe(self):
        for ant in self:
            ant._colony = self

    def transition(self, *args, **kwargs):
        self.move(n_steps=self.n_steps)
        self.update_pheromone()
        for ant in self.get_worst_individuals(0.3):
            ant.reset()

    @property
    def move_proba(self):
        return self.pheromone**self.alpha / np.maximum(self.distances**self.beta, 0.1)

    def init(self):
        for ant in self:
            ant.init()

    def move(self, *args, **kwargs):
        for ant in self:
            ant.move(self, *args, **kwargs)

    def update_pheromone(self):
        delta = np.zeros_like(self.pheromone)
        for ant in self:
            for (i, j) in zip(ant.new_path[:-1], ant.new_path[1:]):
                delta[i, j] += ant.pheromone
        self.pheromone = (1- self.volatilization) * self.pheromone + delta
