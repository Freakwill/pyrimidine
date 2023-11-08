#!/usr/bin/env python3

from .meta import *
from collections import UserString


class C(metaclass=MetaContainer):
    element_class = UserString
    # alias = {'strings': 'elements'}

    def foo(self):
        pass

    def after_setter(self):
        self.fitness = None


def TestMeta():

    def test_attr(self):
        c = C([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        C.set_methods(n_elems=lambda c: 1)
        assert (c.elements == c.strings == ['I', 'love', 'you'] and
                c.lasting == 'for ever' and
                c.n_elements == c.n_strings and
                c.n_elems() == 1)

    def test_method(self):
        c.regester_map('upper')
        assert list(c.upper())==['I', 'LOVE', 'YOU']

        def n_vowels(s):
            return len([o for o in s if str(o) in 'ieaouIEAOU'])
        c.regester_map('length', n_vowels)
        assert list(c.length()) == [1, 2, 2]


    def test_subclass(self):
        class D(C):
            fitness = 1

        d = D([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        d.strings += [UserString('wow')]
        assert d.elements == d.strings
        assert d.fitness is None
