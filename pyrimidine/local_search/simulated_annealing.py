#!/usr/bin/env python3


"""
Simulated Annealing Algorithm

*Ref*
S. Kirkpatrick, C. D. Gelatt, Jr., M. P. Vecchi. Optimization by Simulated Annealing. 1983: 220(4598): 671-679
"""


from .. import PhantomIndividual
from .. import metropolis_rule



class SimulatedAnnealing(PhantomIndividual):
    """Class for Simulated Annealing
    
    Attributes:
        params (dict): parameters in SA
        phantom: phantom solution for exploration
    """

    phantom = None

    params = {'ext_c': 0.99,  # external coef
        'int_c': 0.99,        # internal coef
        'n_epochs': 200,
        'initT': 100,         # initial temperature
        'termT': 0.0001       # terminal temperature
        }

    def init(self):
        # initialize phantom solution
        self.phantom = self.clone(fitness=None)


    def transit(self, *args, **kwargs):
        T = self.initT
        for epoch in range(self.n_epochs):
            self.move(T)
            T *= self.int_c
            if T < self.termT:
                break
        # set the phantom to be the true solution (if it is better then the previous record)
        self.backup()
        self.initT = T * self.ext_c


    def move(self, T):
        """Move phantom
        
        Arguments:
            T {number} -- temperature
        """

        cpy = self.phantom.get_neighbour()

        # Metropolis rule
        flag = metropolis_rule(D=cpy.fitness - self.phantom.fitness, T=T)
        if flag:
            self.phantom.chromosomes = cpy.chromosomes
            self.phantom.fitness = cpy.fitness

