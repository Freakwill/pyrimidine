#!/usr/bin/env python3

from pyrimidine import IterativeMixin, CollectiveMixin
from pyrimidine import MetaContainer


class TestMeta:
    
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
            strategy = 0

        class Game(CollectiveMixin, metaclass=MetaContainer):

            element_class = Player

            @classmethod
            def random(cls):
                return cls(strategy=randint(0, 2), score=0)

            def clone(self, *args, **kwargs):
                return self.__class__(self.strategy, self.score)

            def transition(self):
                pass

        game = Game.random()



    