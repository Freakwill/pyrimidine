#!/usr/bin/env python3

from pyrimidine import IterativeModel


class TestMeta:
    
    def test_iteration(self):
        class TuringModel(IterativeModel):
            pass

        tm = TuringModel()
        assert True
