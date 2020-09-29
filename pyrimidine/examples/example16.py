
from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random()

class _Individual(BinaryIndividual):
    element_class = BinaryChromosome, BinaryChromosome, FloatChromosome


    def decode(self):
        return self[0]

    def _fitness(self):
        return evaluate(self.decode())


from utils import *
class MyPopulation(SGAPopulation):
    element_class = _Individual

    def mate(self):
        male_fits = np.array([ind.fitness for ind in self if ind[-1][0]==1])
        ks = np.argsort(male_fits)
        males = males[ks]
        female_fits = np.array([ind.fitness for ind in self if ind[-1][0]==0])
        ks = np.argsort(female_fits)
        females = females[ks]
        M = len(males)
        F = len(females)

        for k, m in enumerate(males):
            r = randint(0, F-1) / F
            if m[2] <= r:
                f = boltzmann_select(females, female_fits)
                if k / M >=f[2]:
                    m.mate(f)


