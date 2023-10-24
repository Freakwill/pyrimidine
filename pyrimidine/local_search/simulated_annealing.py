#!/usr/bin/env python3


"""
Simulated Annealing Algorithm

*Ref*
S. Kirkpatrick, C. D. Gelatt, Jr., M. P. Vecchi. Optimization by Simulated Annealing. 1983: 220(4598): 671-679
"""


from .. import FitnessModel
from .. import metropolis_rule



class SimulatedAnnealing(FitnessModel):
    """Class for Simulated Annealing
    
    Attributes:
        params (dict): parameters in SA
        phantom: phantom solution for exploration
    """

    phantom = None

    params = {'ext_c': 0.995,
        'int_c': 0.996,
        'nepoch': 200,
        'initT': 100,      # initial temperature
        'termT': 0.0001    # terminal temperature
        }

    def init(self):
        self.phantom = self.clone(fitness=None)

    def transit(self, *args, **kwargs):
        T = self.initT
        for epoch in range(self.nepoch):
            self.phantom.move(T)
            T *= self.int_c
            if T < self.termT:
                break
        if self.fitness < self.phantom.fitness:
            self.chromosomes = self.phantom.chromosomes
            self.fitness = self.phantom.fitness

        self.initT = T * self.ext_c


    def move(self, T):
        """Transition of states
        
        Arguments:
            T {number} -- temperature
        """

        cpy = self.get_neighbour()

        # Metropolis rule
        flag = metropolis_rule(D=cpy.fitness - self.fitness, T=T)
        if flag:
            self.chromosomes = cpy.chromosomes
            self.fitness = cpy.fitness

