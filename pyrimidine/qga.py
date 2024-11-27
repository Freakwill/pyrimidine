#!/usr/bin/env python3


from .chromosome import QuantumChromosome
from .population import HOFPopulation

from .deco import basic_memory


@basic_memory
class _Individual(QuantumChromosome):
    """
    Chromosome/Individual for quantum QA
    """

    def mutate(self):
        # no mutation operation
        pass

    def cross(self, other):
        return self.__class__((self + other) /2)


class QuantumPopulation(HOFPopulation):
    """
    Population for quantum QA
    """

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
