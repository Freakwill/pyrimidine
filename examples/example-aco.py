#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyrimidine.aco import *

from pyrimidine.benchmarks.optimization import *

_evaluate = ShortestPath.random(30)

import numpy as np
distances = np.array([[0, 1, 2, 10], [1,0,10,1], [2,10,0,2], [10, 1,2,0]])

class Ant(BaseAnt): 
    pass
    

class AntPopulation(BaseAntColony):
    element_class = Ant

ap = AntPopulation.from_distances(n_ants=10, distances=distances)
ap.ezolve()

for ant in ap:
    print(ant.path, ant.fitness)