#!/usr/bin/env python3

"""Parallel computing for GAs
"""



import threading


class ThreadWithResult(threading.Thread):
    
    def __init__(self, func, args):
        super().__init__() 
        self._result = None
        self._func = func
        self._args = args

    def run(self):
        self._result = self._func(*self._args)

    @property
    def result(self):
        return self._result


class MTapply:
    # The multi-threading version of the method `apply`

    def __init__(self, type_=None):
        self.type_ = type_

    def __get__(self, instance, owner):
        def _map(f, type_=None):
            threads = [ThreadWithResult(f, (e,)) for e in instance]
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


class MTmap:
    # The multi-threading version of the inbuilt function `map`

    def __init__(self, type_=None):
        self.type_ = type_

    def __call__(self, f, obj, type_=None):
        threads = [ThreadWithResult(f, (e,)) for e in obj]
        for p in threads:
            p.start()
        for p in threads:
            p.join()
        type_ = type_ or self.type_
        if type_ is None:
            return (thread.result if hasattr(thread, 'result') else None for thread in threads)
        else:
            return type_([thread.result if hasattr(thread, 'result') else None for thread in threads])


if __name__ == '__main__':
    
    class C:
        apply = MTapply(type_=list)

        def __init__(self, v=[1,2,3]):
            self.v = v
            self._map = MTmap(type_=tuple)

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
    print(c.map(lambda x:x+1, c))
    print(c.apply(lambda x:x+2))
    c.map = MTmap(type_=tuple)
    print(c.map)
    print(c.map(lambda x:x+1, c))
    c.map = map
    print(c.map)
    print(tuple(c.map(lambda x:x+1, c)))