#!/usr/bin/env python3

import numpy as np

def test_import():
    from pyrimidine import MonoIndividual, BinaryChromosome, StandardPopulation
    from pyrimidine import BaseIndividual, BaseChromosome, BasePopulation
    from pyrimidine.local_search import BaseTabuSearch
    from pyrimidine.benchmarks.special import rosenbrock

    assert isinstance(rosenbrock([1,2,3,4]), (float, np.float_))
