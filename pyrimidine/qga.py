#!/usr/bin/env python3


from .chromosome import QuantumChromosome

from pyrimidine.deco import add_memory


@add_memory({'fitness':None, 'solution':None})
class _Individual(QuantumChromosome):

    def mutate(self):
        pass

    def cross(self, other):
        return self.__class__((self + other) /2)

    def backup(self, check=True):
        f = self._fitness()
        if not check or (self.memory['fitness'] is None or f > self.memory['fitness']):
            self.set_memory(solution=self.solution, fitness=f)


class QuantumPopulation(HOFPopulation):

    element_class = _Individual
    default_size = 20

    def init(self):
        for i in self: i.backup()
        super().init()

    def update_hall_of_fame(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """

        for i in self: i.backup()
        super().update_hall_of_fame(*args, **kwargs)