#!/usr/bin/env python3

from collections import UserString

from pyrimidine.meta import MetaContainer


class TestMeta:

    def setup_method(self):
        class C(metaclass=MetaContainer):
            element_class = UserString
            alias = {'strings': 'elements',
            'n_strings': 'n_elements'}

            def foo(self):
                pass

            def after_setter(self):
                self.fitness = None

        c = C([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        C.set(n_elems=lambda c: 1)
        self.C = C
        self.c = c

    def test_attr(self):
        c = self.c
        assert (c.elements == c.strings == ['I', 'love', 'you'] and
                c.lasting == 'for ever' and
                c.n_elements == c.n_strings and
                c.n_elems() == 1)

    def test_method(self):
        c = self.c
        c.regester_map('upper')
        assert list(c.upper())==['I', 'LOVE', 'YOU']

        def n_vowels(s):
            return len([o for o in s if str(o) in 'ieaouIEAOU'])
        c.regester_map('length', n_vowels)
        assert list(c.length()) == [1, 2, 2]

    def test_subclass(self):
        C = self.C
        class D(C):
            fitness = 1

        d = D([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        d.strings += [UserString('wow')]
        assert d.elements == d.strings
        assert d.fitness is None
