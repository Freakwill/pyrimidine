#!/usr/bin/env python3

from pyrimidine import IterativeMixin, CollectiveMixin


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

    