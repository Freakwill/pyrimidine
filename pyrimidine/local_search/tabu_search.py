#!/usr/bin/env python3

"""
Tabu Search was created by Fred W. Glover in 1986 and formalized in 1989

*References*
Glover, Fred W. and Manuel Laguna. “Tabu Search.” (1997).
Glover, Fred W.. “Tabu Search - Part I.” INFORMS J. Comput. 1 (1989): 190-206.
Glover, Fred W.. “Tabu Search - Part II.” INFORMS J. Comput. 2 (1989): 4-32.
"""

from ..base import BaseIndividual
from ..utils import random, choice
from ..deco import basic_memory


@basic_memory
class BaseTabuSearch(BaseIndividual):
    """Tabu Search algorithm
    """

    params = {'value': 0,
        'tabu_list': [],
        'actions': [],
        'tabu_size': 10
        }

    def transition(self, *args, **kwargs):
        action = choice(self.actions)
        cpy = self.get_neighbour(action)
        if action not in self.tabu_list:
            if cpy.fitness > self.fitness:
                self.chromosomes = cpy.chromosomes
                self.set_memory({
                    'fitness': cpy.fitness,
                    'solution': cpy.decode()
                    })
            else:
                if random() < 0.02:
                    self.chromosomes = cpy.chromosomes
                    self.set_memory({
                        'fitness': cpy.fitness,
                        'solution': cpy.decode()
                        })
                else:
                    self.tabu_list.append(action)
        else:
            if cpy.fitness > self.fitness:
                self.chromosomes = cpy.chromosomes
                self.set_memory({
                    'fitness': cpy.fitness,
                    'solution': cpy.decode()
                    })
                self.tabu_list.remove(action)
        self.update_tabu_list()

    def update_tabu_list(self):
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)

    def get_neighbour(self, action):
        raise NotImplementedError


class SimpleTabuSearch(BaseTabuSearch):

    def get_neighbour(self, action):
        cpy = self.copy()
        i, j = action
        cpy.chromosomes[i][j] = cpy.gene.random()
        return cpy
