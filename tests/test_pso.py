#!/usr/bin/env python3


from pyrimidine.utils import FloatChromosome, BasePopulation
from pyrimidine.pso import Particle, ParticleSwarm
from pyrimidine.benchmarks.special import rosenbrock


def evaluate(x):
    return - rosenbrock(x)


class TestPSO:

    def test_pso(self):
        # generate a knapsack problem randomly

        class _Particle(Particle):
            element_class = FloatChromosome // 8

            def _fitness(self):
                return evaluate(self.position)

        class MyParticleSwarm(ParticleSwarm, BasePopulation):

            element_class = _Particle
            default_size = 10

        ParticleSwarm = MyParticleSwarm

        pop = ParticleSwarm.random()
        pop.transition()
        
        assert isinstance(pop, ParticleSwarm)

