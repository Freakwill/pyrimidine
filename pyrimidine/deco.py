#!/usr/bin/env python3

"""
Decorators
"""

from types import MethodType
from operator import methodcaller
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
        # clear the cache after calling the method
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


usual_side_effect = ['mutate', 'extend', 'pop', 'remove', '__setitem__', '__setattr__', '__setstate__']

def method_cache(func, a):
    """cache for methods

    Pre-define `_cache` as an attribute of the obj.
    
    Args:
        func (TYPE): the original method
        a (TYPE): an attribute (or any value) computed by the method
    
    Returns:
        MethodType
    """
    def mthd(obj):
        # get the attribute from cache, otherwise compute it again
        if obj._cache[a] is None:
            f = obj.func()
            obj._cache[a] = f
            return f
        else:
            return obj._cache[a]
    return mthd


class add_cache:

    """Handle with cache for class
    
    Attributes:
        attrs (tuple[str]): a tuple of attributes
        methods (tuple[str]): a tuple of method names
    """

    def __init__(self, attrs, methods=(), scope=None):
        self.methods = methods
        self.attrs = attrs
        self.scope = scope

    def __call__(self, cls):

        # add `_cache` to the class
        if hasattr(cls, '_cache'):
            cls._cache.update({a: None for a in self.attrs})
        else:
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

        cls.cleared = _cleared
        cls.clear_cache = _clear_cache
        cls.set_cache = _set_cache

        if hasattr(cls, 'copy'):
            cls_copy = cls.copy
            def _copy(obj, cache=True, *args, **kwargs):
                cpy = cls_copy(obj, *args, **kwargs)
                if cache:
                    cpy.set_cache(**obj._cache)
                return cpy
            cls.copy = _copy

        for a in self.attrs:
            if not hasattr(cls, a):
                raise AttributeError(f'The attribute "{a}" should be used in the algorithm!')
            def f(obj):
                """get the attribute from cache, 
                otherwise compute it again by the default method
                """
                if obj._cache[a] is None:      
                    v = getattr(super(cls, obj), a)
                    obj._cache[a] = v
                    return v
                else:
                    return obj._cache[a]
            setattr(cls, a, property(f))

        def _after_setter(obj):
            obj.clear_cache()

        cls.after_setter = _after_setter

        for name in self.methods:
            if hasattr(cls, name):
                setattr(cls, name, clear_cache(getattr(cls, name)))
            else:
                print(f'the class "{cls.__name__}" does not have method "{name}"')

        cls_new = cls.__new__
        def _new(cls, *args, **kwargs):
            obj = cls_new(cls, *args, **kwargs)
            obj._cache = copy.copy(cls._cache)
            return obj

        cls.__new__ = _new

        return cls


fitness_cache = add_cache(('fitness',))


class set_fitness:

    def __init__(self, f=None):
        self.f = f

    def __call__(self, cls):
        if self.f is None:
            if '_fitness' in globals():
                self.f = globals()['_fitness']
            else:
                raise Exception('Function `_fitness` is not defined before setting fitness. You may forget to create the class in the context of environment.')
        cls._fitness = self.f
        return cls


class add_memory:

    """
    add the `_memory` dict to the cls/obj

    The memory dict stores the best solution,
    unlike the `_cache` dict which only records the last computing result.
    And it is not affected by the genetic operations.
    """

    def __init__(self, memory={}):
        self._memory = memory

    def __call__(self, cls):

        cls._memory = self._memory

        def memory(obj):
            return obj._memory

        cls.memory = property(memory)

        def _set_memory(obj, **d):
            obj._memory.update(d)

        cls.set_memory = _set_memory

        def fitness(obj):
            # get fitness from memory by default
            if obj.memory['fitness'] is None:
                return obj._fitness()
            else:
                return obj.memory['fitness']

        cls.fitness = property(fitness)

        for a in cls._memory:
            if a == 'fitness': continue
            def f(obj):
                """get the attribute from memory, 
                where the best solution is stored
                """
                if obj._memory[a] is None:      
                    v = getattr(super(cls, obj), a)
                    return v
                else:
                    return obj._memory[a]
            setattr(cls, a, property(f))

        if hasattr(cls, 'copy'):
            cls_copy = cls.copy
            def _copy(obj, *args, **kwargs):
                cpy = cls_copy(obj, *args, **kwargs)
                cpy._memory = copy.copy(obj._memory)
                return cpy
            cls.copy = _copy

        cls_new = cls.__new__
        def _new(cls, *args, **kwargs):
            obj = cls_new(cls, *args, **kwargs)
            obj._memory = copy.deepcopy(cls._memory)
            return obj

        cls.__new__ = _new

        return cls


def basic_memory(cls):
    # special case of `add_memory`

    cls = add_memory({'fitness':None, 'solution': None})(cls)

    def _backup(obj, check=True):
        """Backup the fitness and other information
        
        Args:
            check (bool, optional): check whether the fitness increases.
        """

        f = super(cls, obj).fitness
        if not check or (obj.memory['fitness'] is None or f > obj.memory['fitness']):
            obj.set_memory(fitness=f, solution=obj.solution)
            
    cls.backup = _backup

    def _init(obj):
        obj.backup(check=False)

    cls.init = _init

    return cls


usual_side_effect = ['mutate', 'extend', 'pop', 'remove', '__setitem__', '__setattr__', '__setstate__']

def method_cache(func, a):
    """cache for methods

    Pre-define `_cache` as an attribute of the obj.
    
    Args:
        func (TYPE): the original method
        a (TYPE): an attribute (or any value) computed by the method
    
    Returns:
        MethodType
    """
    def mthd(obj):
        # get the attribute from cache, otherwise compute it again
        if obj._cache[a] is None:
            f = obj.func()
            obj._cache[a] = f
            return f
        else:
            return obj._cache[a]
    return mthd


class regester_map:

    def __init__(self, mappings, map_=map):
        """
        Args:
            mappings (str, tuple of str): a mapping or mappings
            map_ (None, optional): Description
        
        Raises:
            Exception: Description
        """

        if mappings:
            self.mappings = mappings
        else:
            raise Exception('Have not provided any mapping')
        self.map = map_

    def __call__(self, cls, map_=None):

        if map_ is None:
            if self.map is None:
                if hasattr(cls, map):
                    _map = cls.map
                else:
                    _map = map
            else:
                _map = self.map
        else:
            _map = map_

        if isinstance(self.mappings, str):
            m = self.mappings
            def _m(obj, *args, **kwargs):
                return _map(methodcaller(m, *args, **kwargs), obj)
            setattr(cls, m, _m)
        elif isinstance(self.mappings, tuple):
            for m in self.mappings:
                def _m(obj, *args, **kwargs):
                    return _map(methodcaller(m, *args, **kwargs), obj)
                setattr(cls, m, _m)
        else:
            raise TypeError('`mappings` has to be an instance of str or a tuple of strings')

        return cls


class Regester:
    # regerster operators, used in the future version

    def __init__(name, key=None):
        self.name = name
        self.key = key

    def __call__(self, cls):

        def _regester_operator(obj, name, key=None):
            if hasattr(obj, name):
                raise AttributeError(f'"{name}" is an attribute of "{obj.__name__}", and would not be regestered.')
            obj._operators.append(key)
            setattr(obj, name, MethodType(key, obj))

        def _element_regester(obj, name, e):
            if hasattr(obj, e):
                raise AttributeError(f'"{e}" is an attribute of "{obj.__name__}", would not be regestered.')
            obj._elements.append(e)
            setattr(obj, name, e)

        cls.regester_operator = _regester_operator
        cls.regester_element = _regester_element

        return cls
