#!/usr/bin/env python3

"""
Decorators
"""

from types import MethodType

import copy


def clear_cache(func):
    def mthd(obj, *args, **kwargs):
        result = func(obj, *args, **kwargs)
        obj.clear_cache()
        return result
    return mthd

def side_effect(func):
    """Decorator for methods with side effect

    Apply the decorator to methods with side effects.
    If all the methods called by a particular method have the decorator applied,
    it is not advisable to include the decorator in that method.
    
    Args:
        func (TYPE): a method
    
    Returns:
        Decorated method
    """

    def mthd(obj, *args, **kwargs):
        result = func(obj, *args, **kwargs)
        if hasattr(obj, '_cache'):
            obj.clear_cache()
        return result
    return mthd


def clear_fitness(func):
    def mthd(obj, *args, **kwargs):
        result = func(obj, *args, **kwargs)
        obj.clear_cache('fitness')
        return result
    return mthd


class add_memory:

    def __init__(self, memory={}):
        self._memory = memory

    def __call__(self, cls):

        cls._memory = self._memory

        def memory(obj):
            return obj._memory

        cls.memory = property(memory)

        def fitness(obj):
            if obj.memory['fitness'] is not None:
                return obj.memory['fitness']
            else:
                return obj._fitness()

        cls.fitness = property(fitness)

        cls_clone = cls.clone
        def _clone(obj, *args, **kwargs):
            cpy = cls_clone(obj, *args, **kwargs)
            cpy._memory = obj._memory
            return cpy
        cls.clone = _clone

        cls_new = cls.__new__
        def _new(cls, *args, **kwargs):
            obj = cls_new(cls, *args, **kwargs)
            obj._memory = copy.copy(cls._memory)
            return obj

        cls.__new__ = _new

        return cls


usual_side_effect = ['mutate', 'extend', 'pop', 'remove']


class add_cache:

    """Handle with cache for class
    
    Attributes:
        attrs (tuple[str]): a tuple of attributes
        methods (tuple[str]): a tuple of method names
    """

    def __init__(self, attrs, methods=()):
        self.methods = methods
        self.attrs = attrs

    def __call__(self, cls):
        cls._cache = {a: None for a in self.attrs}

        @property
        def cache(obj):
            return obj._cache

        cls.cache = property(cache)

        def _clear_cache(obj, k=None):
            if k is None:
                obj._cache = {k: None for k in obj._cache.keys()}
            elif k in obj._cache:
                obj._cache[k] = None

        def _cleared(obj, k=None):
            if k is None:
                return all(v == None for v in obj._cache.values())
            elif k in obj._cache:
                return obj._cache[k] == None

        def _set_cache(obj, **d):
            obj._cache.update(d)

        cls_clone = cls.clone
        def _clone(obj, cache=True, *args, **kwargs):
            cpy = cls_clone(obj, *args, **kwargs)
            if cache:
                cpy.set_cache(**obj._cache)
            return cpy

        cls.cleared = _cleared
        cls.clear_cache = _clear_cache
        cls.set_cache = _set_cache
        cls.clone = _clone

        for a in self.attrs:
            def f(obj):
                if obj._cache[a] is None:
                    f = getattr(obj, '_'+a)()
                    obj._cache[a] = f
                    return f
                else:
                    return obj._cache[a]
            setattr(cls, a, property(f))

        def _after_setter(obj):
            obj.clear_cache()

        cls.after_setter = _after_setter

        for name in self.methods:
            if hasattr(obj, name):
                setattr(cls, name, clear_cache(getattr(cls, name)))

        cls_new = cls.__new__
        def _new(cls, *args, **kwargs):
            obj = cls_new(cls, *args, **kwargs)
            obj._cache = copy.copy(cls._cache)
            return obj

        cls.__new__ = _new

        return cls


fitness_cache = add_cache(('fitness',))


class Regester:
    # regerster operators, used in the future version

    def __init__(name, key=None):
        self.name = name
        self.key = key

    def __call__(cls):

        def _regester_operator(self, name, key=None):
            if hasattr(self, name):
                raise AttributeError(f'"{name}" is an attribute of "{self.__name__}", and would not be regestered.')
            self._operators.append(key)
            setattr(self, name, MethodType(key, self))

        def _element_regester(self, name, e):
            if hasattr(self, e):
                raise AttributeError(f'"{e}" is an attribute of "{self.__name__}", would not be regestered.')
            self._elements.append(e)
            setattr(self, name, e)

        cls.regester_operator = _regester_operator
        cls.regester_element = _regester_element

        return cls
