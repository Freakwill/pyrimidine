#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from random import random

class TraitIndividual(MixIndividual):
    element_class = BinaryChromosome, FloatChromosome

    def mutate(self):
        if random() < self.mutate_prob:
            super(TraitIndividual, self).mutate()

    def mate(self, other, mate_prob=None):
        if random() < (mate_prob or self.mate_prob):
            return super(TraitIndividual, self).mate(other)
        else:
            return self

    @property
    def mutate_prob(self):
        return self.chromosomes[-1][0]

    @property
    def mate_prob(self):
        return self.chromosomes[-1][1]


class TraitThresholdIndividual(TraitIndividual):
    ranking = None

    def mate(self, other, mate_prob=None):
        if other.ranking:
            if self.threshold <= other.ranking:
                return super(TraitThresholdIndividual, self).mate(other, mate_prob=1)
            elif self.threshold <= 2* other.ranking:
                return super(TraitThresholdIndividual, self).mate(other, mate_prob=0.8)
            else:
                return super(TraitThresholdIndividual, self).mate(other, mate_prob=0.5)
        else:
            return super(TraitThresholdIndividual, self).mate(other)

    @property
    def threshold(self):
        return self.chromosomes[-1][-1]


