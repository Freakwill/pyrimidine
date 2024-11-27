#!/usr/bin/env python3

import numpy as np
 

def test_import():
    from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
    from pyrimidine.base import BaseIndividual, BaseChromosome, BasePopulation
    from pyrimidine.local_search import BaseTabuSearch
    from pyrimidine.benchmarks.special import rosenbrock

    assert isinstance(rosenbrock([1,2,3,4]), np.number)


def test_import_sa():
    from pyrimidine.local_search import SimulatedAnnealing
    from pyrimidine.base import BaseIndividual

    assert issubclass(SimulatedAnnealing, BaseIndividual)
