#!/usr/bin/env python3


from .chromosome import QuantumChromosome

from .deco import basic_memory

@basic_memory
class _Individual(QuantumChromosome):

    def mutate(self):
        pass

    def cross(self, other):
        return self.__class__((self + other) /2)


class QuantumPopulation(HOFPopulation):

    element_class = _Individual
    default_size = 20

    def backup(self):
        for i in self: i.backup()

    def update_hall_of_fame(self, *args, **kwargs):
        """
        Update the `hall_of_fame` after each step of evolution
        """

        self.backup()
        super().update_hall_of_fame(*args, **kwargs)
