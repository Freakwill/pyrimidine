#!/usr/bin/env python3

"""Particle Swarm Optimization

Developed by J. Kennedy and R. Eberhart[Kennedy and Eberhart 2001]

Each individual is represented position and velocity.
"""

from .base import PopulationModel
from .chromosome import FloatChromosome
from .individual import MemoryIndividual
from .utils import gauss, random
from operator import attrgetter

import numpy as np

class BaseParticle:
    """A particle in PSO
    
    Extends:
        PolyIndividual
    
    Variables:
        default_size {number} -- one individual represented by 2 chromosomes: position and velocity
        memory {Particle} -- the best state of the particle moving in the solution space.

    Caution:
        Have to implement the properties in subclasses: position, velocity
    """

    element_class = FloatChromosome
    default_size = 2

    _memory = {'position': None,
            'fitness': None}    # store the best position passed by the particle

    params = {'learning_factor': 2,
            'acceleration_coefficient': 3,
            'inertia': 0.5
            }


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
        return self.memory['position']

    def update_vilocity(self, fame=None, *args, **kwargs):
        raise NotImplementedError

    def move(self):
        self.position += self.velocity

    def decode(self):
        return self.best_position


class Particle(BaseParticle, MemoryIndividual):

    element_class = FloatChromosome
    default_size = 2

    @property
    def position(self):
        return self.chromosomes[0]

    @position.setter
    def position(self, x):
        self.chromosomes[0] = x
        self.__fitness = None

    @property
    def velocity(self):
        return self.chromosomes[1]

    @velocity.setter
    def velocity(self, v):
        self.chromosomes[1] = v

    def update_vilocity(self, fame=None, *args, **kwargs):
        xi = random()
        if fame is None:
            particle.velocity = (self.inertia * particle.velocity
                + self.learning_factor * xi * (particle.best_position-particle.position))
        else:
            eta = random()
            particle.velocity = (self.inertia * particle.velocity
                 + self.learning_factor * xi * (particle.best_position-particle.position)
                 + self.acceleration_coefficient * eta * (fame.best_position-particle.position))


class ParticleSwarm(PopulationModel):
    """Standard PSO
    
    Extends:
        PopulationModel
    """
    
    element_class = Particle
    default_size = 20

    params = {'learning_factor': 2, 'acceleration_coefficient': 3,
    'inertia':0.5, 'n_best_particles':0.2, 'max_velocity':None}

    def init(self):
        for particle in self:
            particle.init()

        self.hall_of_fame = self.get_best_individuals(self.n_best_particles, copy=True)

    
    def update_hall_of_fame(self):
        hof_size = len(self.hall_of_fame)
        for ind in self:
            for k in range(hof_size):
                if self.hall_of_fame[-k-1].fitness < ind.fitness:
                    self.hall_of_fame.insert(hof_size-k, ind.clone())
                    self.hall_of_fame.pop(0)
                    break


    @property
    def best_fitness(self):
        if self.hall_of_fame:
            return np.max(list(map(attrgetter('fitness'), self.hall_of_fame)))
        else:
            return super().best_fitness
    

    def transition(self, *args, **kwargs):
        """
        Transitation of the states of particles
        """
        self.move()
        self.backup()
        self.update_hall_of_fame()


    def backup(self):
        # overwrite the memory of the particle if its current state is better its memory
        for particle in self:
            particle.backup(check=True)


    def move(self):
        """moving rule of particles

        Particles move according to the hall of fame and the best record
        """
        xi = random()
        eta = random()
        for particle in self:
            if particle in self.hall_of_fame:
                particle.velocity = (self.inertia * particle.velocity
             + self.learning_factor * xi * (particle.best_position-particle.position))
            else:
                for fame in self.hall_of_fame:
                    if particle.fitness < fame.fitness:
                        break
                particle.velocity = (self.inertia * particle.velocity
                 + self.learning_factor * xi * (particle.best_position-particle.position)
                 + self.acceleration_coefficient * eta * (fame.best_position-particle.position))
            particle.position = particle.position + particle.velocity



class DiscreteParticleSwarm(ParticleSwarm):
    def move(self):
        """moving rule of particles

        Particles move according to the hall of fame and the best record
        """
        v1 = self.inertia
        v2 = self.learning_factor
        v3 = self.acceleration_coefficient
        for particle in self:
            if particle in self.hall_of_fame:
                particle.velocity = def_velocity1(particle, v1, v2)
            else:
                for fame in self.hall_of_fame:
                    if particle.fitness < fame.fitness:
                        break
                particle.velocity = def_velocity2(particle, fame, v1, v2, v3)
            particle.move()
