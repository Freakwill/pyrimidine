#!/usr/bin/env python3


from pyrimidine.deco import *


class TestDeco:

    def test_cache(self):

        @add_cache(attrs=('f',), methods=('bar',))
        class C:
            v = 1
            def _f(self):
                return self.v

            def bar(self):
                pass

        c = C()

        assert c._f() == 1
        assert c.f == 1
        assert c._cache['f'] == 1
        c.v = 2
        assert c._f() == 2
        c.bar()
        assert c.f == 2

    def test_set_fitness(self):

        _fitness = lambda x: x.v+1

        @set_fitness()
        class C:
            v = 2

        c = C()
        assert c._fitness() == 3

    def test_map(self):

        @regester_map('upper')
        class C:
            elms = ['a', 'b']

            def __iter__(self):
                return iter(self.elms)

        c = C()

        assert list(c.upper()) == ['A', 'B']

