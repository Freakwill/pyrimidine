#!/usr/bin/env python3

"""
Sparrow Search Algorithm

The framework of the SSA

    Input: 
    G: the maximum iterations 
    PD: the number of producers 
    SD: the number of sparrows who perceive the danger R2 : the alarm value 
    n: the number of sparrows 
    Initialize a population of n sparrows and define its relevant parameters. 
    Output: Xbest, fg. 
    while (t < G) 
        Rank the fitness values and find the current best individual and the current worst individual. 
        R2 = rand(1)
        for i = 1 : PD
            update the sparrow’s location;
        for i = (PD + 1) : n
            update the sparrow’s location;
        for l = 1 : SD
            update the sparrow’s location;
        Get the current new location;
        If the new location is better than before, update it;
        t = t + 1
    return Xbest, fg.

*References*
Jiankai Xuea, and Bo Shena, A novel swarm intelligence optimization approach: sparrow search algorithm.
"""

from random import gauss, random, randint
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..mixin import PopulationMixin
from ..chromosome import FloatChromosome

from ..deco import basic_memory


@basic_memory
class BaseSparrow(FloatChromosome):

    def move(self, ST, i):
        r = random()
        Q = gauss()
        alpha = random()+0.01
        if r<ST:
            self *= np.exp(-i/(T*alpha))
        else:
            self += Q


class Producer(BaseSparrow):
    pass


class Scrounger(BaseSparrow):

    def move(self, worst, producer, i):
        r = random()
        if r < 0.5:
            Q = gauss()
            self[:] = Q * np.exp((self - worst) / i**2)
        else:
            d = len(self)
            self[:] = producer + np.abs(self - producer) * (np.random.randint(2, size=d)*2 - 1) /d**2


class StandardSparrowSearch(PopulationMixin):
    """Starndard Sparrow Search Algorithm
    """

    element_class = BaseSparrow

    params = {
        "PD": 0.2,
        "SD": 0.2,
        "ST": 0.7,
        'rho': 0.001
    }

    def init(self):
        if 0< self.PD <1:
            self.PD = int(self.PD * len(self))
        if 0< self.SD <1:
            self.SD = int(self.SD * len(self))

    def transition(self, *args, **kwargs):

        self.sort()

        producers = self[:self.PD]
        
        for i, sparrow in enumerate(producers):
            sparrow.move(self.ST, i)

        k = np.argmax(list(map(lambda x: x.fitness, producers)))
        producer = producers[k]

        for i, sparrow in enumerate(self[self.PD:]):
            Scrounger.move(sparrow, self[0], producer, i+self.PD)

        best, worst = self[-1], self[0]
        bf, wf = self.max_fitness, self.min_fitness

        for sparrow in self.random_select(n_sel=self.SD):
            if sparrow.fitness < bf:
                beta = gauss()
                sparrow[:] = best + beta * np.abs(best - sparrow)
            elif sparrow.fitness == bf:
                K = randint(0, 1)*2 - 1
                sparrow += K * np.abs(sparrow - worst) / (sparrow.fitness - wf + self.rho)

        self.update()

