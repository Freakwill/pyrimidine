#!/usr/bin/env python3

from random import randint

from pyrimidine import IterativeMixin, CollectiveMixin
from pyrimidine import MetaContainer


class TestMixin:
    
    def test_iterative(self):
        class TuringMachine(IterativeMixin):
            pass

        tm = TuringMachine()
        assert True

    def test_collective(self):
        class DoubleTuringMachine(CollectiveMixin):
            pass

        dtm = DoubleTuringMachine()
        assert True

    def test_game(self):

        class Player:

            def __init__(self, strategy=0, score=0):
                self.strategy = strategy # 1,2
                self.score = score

            def copy(self, *args, **kwargs):
                return self.__class__(self.strategy, self.score)

            @classmethod
            def random(cls):
                return cls(strategy=randint(0, 2), score=0)


        class Game(CollectiveMixin, metaclass=MetaContainer):

            element_class = Player
            default_size = 10

            def transition(self):
                pass

        game = Game.random()
        game.save(filename='model.pkl')
        assert True
        # game_ = Game.load(filename='model.pkl')
        # assert all(p.strategy == p_.strategy for p, p_ in zip(game, game_))
            