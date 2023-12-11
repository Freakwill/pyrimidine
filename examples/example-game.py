#!/usr/bin/env python


from random import random, randint
import numpy as np

from pyrimidine.mixin import CollectiveMixin
from pyrimidine.meta import MetaContainer


class Player:
    """
    Play the "scissors, paper, stone" game

    `scissors`, `paper`, `stone` = 0, 1, 2
    """

    params = {'mutate_prob': 0.1}

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

    def init(self):
        pass

    def __lt__(self, other):
        return ((self.strategy, other.strategy) == (0, 1)
            or (self.strategy, other.strategy) == (1, 2)
            or (self.strategy, other.strategy) == (2, 0))

    def __str__(self):
        return f'{self.strategy}: {self.score}'


class Game(CollectiveMixin, metaclass=MetaContainer):

    params = {'compete_prob': 0.5, 'mutate_prob': 0.2}

    element_class = Player
    default_size = 100

    def transition(self, *args, **kwargs):
        self.compete()
        self.duplicate()
        self.mutate()

    def mutate(self, mutate_prob=None):
        for player in self:
            if random() < (mutate_prob or self.mutate_prob):
                player.mutate()

    def compete(self):
        k = int(0.5 * self.default_size)
        winner = []
        for i, p in enumerate(self[:-1]):
            for j, q in enumerate(self[:i]):
                if random() < self.compete_prob:
                    if p < q:
                        p.score += 1
                        q.score -= 1      
                    elif q < p:
                        p.score -= 1
                        q.score += 1
        winners = np.argsort([p.score for p in self])[-k:]
        self.elements = [self.elements[k] for k in winners]

    def duplicate(self):
        self.extend(self.clone())


game = Game.random()
stat = {'scissors': lambda game: sum(p.strategy==0 for p in game),
'paper': lambda game: sum(p.strategy==1 for p in game),
'stone': lambda game: sum(p.strategy==2 for p in game)
}
data = game.evolve(stat=stat, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['scissors', 'paper', 'stone']].plot(ax=ax)
ax.set_title("Have a zero-sum game")
plt.show()
