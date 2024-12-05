#!/usr/bin/env python3

"""
Parallel computing for GAs
"""


import threading
from dask import compute, delayed


class ThreadWithResult(threading.Thread):
    # The tread storing the computing results
    
    def __init__(self, func, *args, **kwargs):
        super().__init__() 
        self._result = None
        self._func = func
        self._args = args
        self._kwargs

    def run(self):
        self._result = self._func(*self._args, **self._kwargs)

    @property
    def result(self):
        return self._result


class MTApply:
    """Descriptor, the multi-threading version of the method `apply`

    Attributes:
        type_ (TYPE): the type of the return value, such as `list`.
                      set to be None, by default.
    """

    def __init__(self, type_=None):
        self.type_ = type_

    def __get__(self, instance, owner):
        def _map(f, type_=None):
            threads = [ThreadWithResult(f, e) for e in instance]
            for p in threads:
                p.start()
            for p in threads:
                p.join()
            type_ = type_ or self.type_
            if type_ is None:
                return (thread.result if hasattr(thread, 'result') else None for thread in threads)
            else:
                return type_([thread.result if hasattr(thread, 'result') else None for thread in threads])
        return _map


class MTMap:
    """The multi-threading version of the inbuilt function `map`
    
    Attributes:
        type_ (TYPE): the type of the return value, such as `list`.
                      set to be None, by default.
    """

    def __init__(self, type_=None):
        self.type_ = type_

    def __call__(self, f, obj, type_=None):
        threads = [ThreadWithResult(f, e) for e in obj]
        for p in threads:
            p.start()
        for p in threads:
            p.join()
        type_ = type_ or self.type_
        if type_ is None:
            return (thread.result if hasattr(thread, 'result') else None for thread in threads)
        else:
            return type_([thread.result if hasattr(thread, 'result') else None for thread in threads])


class DaskApply:
    """Descriptor, the parallel version of the method `apply` by `dask`

    Attributes:
        type_ (TYPE): the type of the return value, such as `list`.
                      set to be None, by default.

    Also see: MTApply
    """

    def __init__(self, type_=None):
        self.type_ = type_

    def __get__(self, instance, owner):
        def _map(f, type_=None):
            results = compute(*(delayed(f)(e) for e in instance))
            type_ = type_ or self.type_
            if type_ is None:
                return results
            else:
                return type_(results)
        return _map


class DaskMap:
    """The parallel version of the inbuilt function `map` by `dask`

    Attributes:
        type_ (TYPE): the type of the return value, such as `list`.
                      set to be None, by default.
    Also see: MTMap
    """

    def __init__(self, type_=None):
        self.type_ = type_

    def __call__(self, f, obj, type_=None):
        results = compute(*(delayed(f)(e) for e in obj))
        type_ = type_ or self.type_
        if type_ is None:
            return results
        else:
            return type_(results)


def mt_apply(cls):
    # The class decorator to make the class's apply method use parallel computing
    cls.apply = MTApply(type_=list)
    return cls


def dask_apply(cls):
    cls.apply = DaskApply(type_=list)
    return cls


if __name__ == '__main__':
    
    @dask_apply
    class C:

        def __init__(self, v=[1,2,3]):
            self.v = v
            self._map = MTMap(type_=tuple)

        @property
        def map(self):
            return self._map

        @map.setter
        def map(self, v):
            self._map = v

        def __iter__(self):
            return iter(self.v)

    c = C([1,2,3,4])
    print(c.map)
    print(c.map(lambda x:x+1, c)==(2, 3, 4, 5))
    print(c.apply(lambda x:x+2))
    c.map = DaskMap(type_=tuple)
    print(c.map)
    print(c.map(lambda x:x+1, c))
    c.map = map
    print(c.map)
    print(tuple(c.map(lambda x:x+1, c)))
    