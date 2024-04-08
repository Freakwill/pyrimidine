#!/usr/bin/env python3


from pyrimidine import FloatChromosome, BasePopulation
from pyrimidine.ep import BaseEPIndividual, EvolutionProgramming
from pyrimidine.benchmarks.special import rosenbrock


def evaluate(x):
    return - rosenbrock(x)


class TestEP:

    def test_ep(self):
        # generate a knapsack problem randomly

        class _Individual(BaseEPIndividual):
            element_class = FloatChromosome // n, FloatChromosome // n

            def decode(self):
                return self.chromosomes[0]


            def _fitness(self):
                return - _evaluate(self.decode())


        class _Population(EvolutionProgramming, BasePopulation):
            element_class = _Individual
            default_size = 20


        pop = _Population.random()
        pop.init()
        pop.transition()
        
        assert isinstance(pop, _Population)

