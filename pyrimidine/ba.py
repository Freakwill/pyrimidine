from .base import *

class Bat(PolyIndividual):
    element_class = FloatChromosome
    default_size = 5
    phantom = None

    def backup(self):
        self.memory = self.clone(fitness=self.fitness)

    def init(self):
        self.memory = self.clone(fitness=self.fitness)

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