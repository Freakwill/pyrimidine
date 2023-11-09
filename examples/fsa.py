#!/usr/bin/env python3


from pyrimidine import System, IterativeModel
from pyrimidine.utils import methodcaller

import numpy as np


class BaseMachine(IterativeModel, metaclass=System):

    def init(self):
        self.state = self.init_state

    def halt(self):
        pass


class FSA(BaseMachine):
    """
    Finite state automata
    """
    init_state = 0

    def halt(self):
        return self.state == -1

    def transit(self, k):
        if self.state == 0:
            if self.inputs[k] == 0:
                self.state = 0
            elif self.inputs[k] == 1:
                self.state = 1
        elif self.state == 1:
            if self.inputs[k] == 0:
                self.state = -1
            elif self.inputs[k] == 1:
                self.state = 2
        elif self.state == 2:
            if self.inputs[k] == 0:
                self.state = -1
            elif self.inputs[k] == 1:
                self.state = 1
        else:
            print('end')


    def evolve(self, *args, **kwargs):
        super().evolve(control=methodcaller('halt'), *args, **kwargs)


fsa = FSA()
fsa.inputs = [0, 1, 1, 1, 1, 1, 0, 1]
fsa.evolve(verbose=True)


# class TuringMachine(BaseMachine):
#     pass
