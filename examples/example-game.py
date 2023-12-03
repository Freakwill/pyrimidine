#!/usr/bin/env python


from random import random, randint

import numpy as np

from pyrimidine import BasePopulation


class Player:

    params = {'mutate_prob': 0.02}

    def __init__(self, strategy=0, score=0):
        self.strategy = strategy # 1,2
        self.score = score

    @classmethod
    def random(cls):
        return cls(strategy=randint(0, 2), score=0)

    def clone(self, *args, **kwargs):
        return self.__class__(self.strategy, self.score)

    def mutate(self):
        self.strategy = randint(0, 2)


class Game(BasePopulation):

    element_class = Player
    default_size = 50

    def transition(self):
        self.compete()
        self.duplicate()
        self.mutate()

    def compete(self):
        k = int(0.5 * self.default_size)
        winner = []
        for i, p in enumerate(self[:-1]):
            for j, q in enumerate(self[:i]):
                if random() < 0.5:
                    if (p.strategy, q.strategy) == (0, 1):
                        p.score -= 1
                        q.score += 1      
                    elif (p.strategy, q.strategy) == (0, 2):
                        p.score += 1
                        q.score -= 1
                    elif (p.strategy, q.strategy) == (1, 2):
                        p.score -= 1
                        q.score += 1
                    elif (p.strategy, q.strategy) == (1, 0):
                        p.score += 1
                        q.score -= 1
                    elif (p.strategy, q.strategy) == (2, 0):
                        p.score -= 1
                        q.score += 1
                    elif (p.strategy, q.strategy) == (2, 1):
                        p.score += 1
                        q.score -= 1
        winners = np.argsort([p.score for p in self])[-k:]
        self.elements = [self.elements[k] for k in winners]

    def duplicate(self):
        self.extend(self.clone())


pop = Game.random()
import collections
c = collections.Counter([i.strategy for i in pop])
print(c)
for _ in range(10):
    pop.transition()
    c = collections.Counter([i.strategy for i in pop])
    print(c)
