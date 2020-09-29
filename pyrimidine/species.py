#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import BaseSpecies

class DualSpecies(BaseSpecies):

    def mate(self):
        for individual0 in self.populations[0]:
            individual0.cross(individual1) for individual1 in self.populations[1]

    def transitate(self, *args, **kwargs):
        pass
