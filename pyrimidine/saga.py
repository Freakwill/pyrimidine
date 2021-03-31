#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from . import *
from .utils import *
from random import random

class BaseSelfAdaptiveIndividual(MixedIndividual):

    @property
    def mutate_prob(self):
        raise NotImplementedError

    @mutate_prob.setter
    def mutate_prob(self, v):
        raise NotImplementedError

    @property
    def mate_prob(self):
        raise NotImplementedError

    @mate_prob.setter
    def mate_prob(self, v):
        raise NotImplementedError

    @property
    def desire(self):
        raise NotImplementedError


class _SelfAdaptiveIndividual(BaseSelfAdaptiveIndividual):
    """Individual for Self-adaptive GA

    Provide at least 2 chromosomes, one of them is coded by float numbers
    representing the prob of mutation and mating.

    Extends:
        BaseSelfAdaptiveIndividual
    """
    
    # By default, the second chromosome (a float vector) encodes the prob of mutation and mating
    element_class = BinaryChromosome, FloatChromosome

    def mutate(self):
        if random() < self.mutate_prob:
            super(SelfAdaptiveIndividual, self).mutate()

    def mate(self, other, mate_prob=None):
        if random() < (mate_prob or (self.mate_prob + other.mate_prob)/2):
            return super(SelfAdaptiveIndividual, self).mate(other)
        else:
            return self

    @property
    def mutate_prob(self):
        return self.chromosomes[-1][0]

    @mutate_prob.setter
    def mutate_prob(self, v):
        self.chromosomes[-1][0] = v

    @property
    def mate_prob(self):
        return self.chromosomes[-1][1]

    @mate_prob.setter
    def mate_prob(self, v):
        self.chromosomes[-1][1] = v

    @property
    def desire(self):
        return self.chromosomes[-1][2]
    

class SelfAdaptiveIndividual(_SelfAdaptiveIndividual):
    ranking = None

    # def mate(self, other, mate_prob=None):
    #     if other.ranking:
    #         if self.threshold <= other.ranking:
    #             return super(TraitThresholdIndividual, self).mate(other, mate_prob=1)
    #         elif self.threshold <= 2* other.ranking:
    #             return super(TraitThresholdIndividual, self).mate(other, mate_prob=0.8)
    #         else:
    #             return super(TraitThresholdIndividual, self).mate(other, mate_prob=0.5)
    #     else:
    #         return super(TraitThresholdIndividual, self).mate(other)

    # @property
    # def threshold(self):
    #     return self.chromosomes[-1][-1]

