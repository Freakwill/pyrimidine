#!/usr/bin/env python3


class BaseProblem:
    """Base Class of Problems
    
    Please implement `evaluate` method in each subclass
    """
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)