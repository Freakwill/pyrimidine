#!/usr/bin/env python3


from pyrimidine.aco import *


import numpy as np

distances = np.array([[0, 1, 2, 10,5], [1,0,10,1,4], [2,10,0,2,3], [10, 1,2,0,1], [5,4,3,1, 0]])


class Ant(BaseAnt): 
    pass


class AntPopulation(BaseAntColony):
    element_class = Ant


ac = AntPopulation.from_distances(n_ants=10, distances=distances)
ac.ezolve(n_iter=30)

print('colony:')
for ant in ac:
    print(ant.path, ant.get_length())

print('solution:')
print(ac.solution.path, ac.solution.get_length(), ac.solution.fitness)
