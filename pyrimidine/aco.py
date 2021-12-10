#!/usr/bin/env python3

from types import MethodType
import numpy as np
from .base import BaseFitnessModel, BasePopulationModel
from .meta import MetaContainer
from scipy.stats import rv_discrete

class BaseAnt(BaseFitnessModel):
    initial_position = 0
    __path = [0]

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, p):
        self.__path = p
        self.fitness = None


    def __init__(self, *args, **kwargs):
        if args:
            raise Exception('__init__ has no position arguments!')
        for k, v in kwargs.items():
            setattr(self, k, v)


    def move(self, colony=None):
        while True:
            i = self.path[-1]
            next_positions = []
            ps = []
            for j in colony.positions:
                if j not in self.path:
                    pij = colony.pheromone[i,j]**colony.alpha/colony.distances[i,j]**colony.beta+0.001
                    ps.append(pij)
                    next_positions.append(j)
            if not next_positions: break
            ps /= np.sum(ps)
            rv = rv_discrete(values=(next_positions, ps))
            self.path.append(rv.rvs(size=1)[0])
        self._fitness = MethodType(lambda o:np.sum(colony.distances[i,j] for i, j in zip(self.path[:-1], self.path[1:])), self)
        self.release_pheromone(colony)

    def release_pheromone(self, colony):
        self.pheromone = colony.sedimentation_constant / self.fitness

    # def _fitness(self):
    #     # length of the path that the ant passed
    #     return self.fitness_


class BaseAntColony(BasePopulationModel, metaclass=MetaContainer):

    element_class = BaseAnt
    params = {'sedimentation_constant':100, 'volatilization_rate':0.75, 'alpha':1, 'beta':5}


    @classmethod
    def from_positions(cls, n_ants=10, positions=None):
        obj = cls(elements=[cls.element_class() for _ in range(n_ants)])
        obj.positions = positions
        obj.pheromone = np.zeros((len(positions),)*2)
        from scipy.spatial.distance import pdist, squareform
        obj.distances = squareform(pdist(points))
        return obj

    @classmethod
    def from_distances(cls, n_ants=10, distances=None):
        obj = cls(elements=[cls.element_class(distances=distances) for _ in range(n_ants)])
        obj.positions = np.arange(distances.shape[0])
        obj.distances = distances
        obj.pheromone = np.zeros(distances.shape)
        return obj


    @property
    def ants(self):
        return self.elements


    def transit(self, *args, **kwargs):
        self.init()
        self.move()
        self.update_pheromone()

    def init(self):
        for ant in self.ants:
            ant.path = [0]

    def move(self, *args, **kwargs):
        for ant in self.ants:
            ant.move(self, *args, **kwargs)

    def update_pheromone(self):
        delta = 0
        for ant in self.ants:
            delta += ant.pheromone
        self.pheromone = (1- self.volatilization_rate)*self.pheromone + delta
