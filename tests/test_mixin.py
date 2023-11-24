#!/usr/bin/env python3

import unittest

from pyrimidine import IterativeModel


class TestMeta(unittest.TestCase):
    
    def test_iteration(self):
        class TuringModel(IterativeModel):
            pass

        tm = TuringModel()
        assert True
