#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import MethodType
from collections.abc import Iterable
from operator import attrgetter


def get_stem(s):
    """get the last part in Camel expression
    
    Arguments:
        s {str} -- a string in Camel expression
    
    Returns:
        str -- last part of s in lower form

    Example:
        >>> get_stem('ILoveYou')
        >>> you
    """

    for k, a in enumerate(s[::-1]):
        if a.isupper(): break
    return s[-k-1:].lower()

from .mixin import *

class ParamType(type):
    """just a wrapper of `type`

    Define `params` in classes whose metaclass of ParamType,
    then key-value pairs in `params` could be inherited from super classes, like attributes.
    It make users set and manage parameters of classes or instances more easily.
    """
    def __new__(cls, name, bases=(), attrs={}):
        # inherit alias instead of overwriting it, when setting `alias` for a subclass
        # alias is not recommended to use!
        alias = {}
        for b in bases:
            if hasattr(b, 'alias') and b.alias:
                alias.update(b.alias)

        alias.update(attrs.get('alias', {}))
        attrs['alias'] = alias

        # inherit params instead of overwriting it, when setting `params` for a subclass
        params = {}
        for b in bases:
            if hasattr(b, 'params') and b.params:
                params.update(b.params)

        params.update(attrs.get('params', {}))
        attrs['params'] = params

        def _getattr(self, key):
            if key in self.params:
                return self.params[key]
            elif key in self.alias:
                return getattr(self, self.alias[key])
            else:
                return self.__getattribute__(key)
        attrs['__getattr__'] = _getattr

        def _setattr(self, key, value):
            if key in self.params:
                self.params[key] = value
            elif key in self.alias:
                setattr(self, self.alias[key], value)
            else:
                super(IterativeModel, self).__setattr__(key, value)
        attrs['__setattr__'] = _setattr

        return super().__new__(cls, name, bases, attrs)


class System(ParamType):
    """Abstract class of systems

    A system consists of a set of elements and operators acting on them

    It is refered to an algebraic system.
    """
    
    def __new__(cls, name, bases, attrs):
        # Create with regesters
        
        def _iter(self):
            return iter(self.__elements)

        def _getitem(self, k):
            # print(DeprecationWarning('get item directly is not recommended now.'))
            return self.__elements[k]

        def _len(self):
            if hasattr(self, '__n_elements'):
                return getattr(self, '__n_elements')
            return len(self.__elements)

        attrs.update(
            {'__getitem__': _getitem,
            '__len__': _len,
            '__iter__': _iter}
        )

        def _get_all(self, attr_name):
            return map(attrgetter(attr_name), self.__elements)

        @property
        def _elements(self):
            return self.__elements

        @_elements.setter
        def _elements(self, x):
            self.__elements = x
            # L = len(x)
            # self.__n_elements = L
            # setattr(self, '__n_' + element_name, L)

        @property
        def _n_elements(self):
            # return self.__n_elements
            return len(self.elements)

        attrs.update(
            {"elements": _elements,
            "n_elements": _n_elements,
            "get_all": _get_all}
        )

        @property
        def _operators(self):
            return self.__operators

        @_operators.setter
        def _operators(self, x):
            self.__operators = x

        attrs.update(
            {"operator": _operators}
        )

        def _type_check(self):
            return all(isinstance(elm, self.element_class) for elm in self.__elements)
        attrs['type_check'] = _type_check

        def _regester_map(self, name, key=None):
            if key is None:
                key = lambda e: getattr(e, name)()
            def m(obj):
                return map(key, obj.elements)
            setattr(self, name, MethodType(m, self))

        attrs['regester_map'] = _regester_map
        
        # def _operator_regester(self, m):
        #     if hasattr(self, m):
        #         raise AttributeError(f'`{m}` is an attribute of {self.__class__.__name__}, would not be regestered.')
        #     def _m(obj, *args, **kwargs):
        #         return [getattr(a, m)(*args, **kwargs) for a in obj]
        #     setattr(self, m, MethodType(_m, self))

        # def _element_regester(self, e):
        #     if hasattr(self, e):
        #         raise AttributeError(f'`{e}` is an attribute of {self.__class__.__name__}, would not be regestered.')
        #     @property
        #     def _p(obj):
        #         return [getattr(a, e) for a in obj]
        #     setattr(self, e, _p)

        # attrs.update({'operator_regester': _operator_regester,
        # 'element_regester': _element_regester})

        return super().__new__(cls, name, bases, attrs)


    def set(self, *args, **kwargs):
        for k in args:
            setattr(self, k, globals()[k])
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


    def set_methods(self, **kwargs):
        for k, m in kwargs.items():
            setattr(self, k, MethodType(m, self))
        return self


    def __call__(self, *args, **kwargs):
        o = super().__call__()

        if not args:
            raise Exception('Did not provide a list of elements as the unique positional argument!')
        else:
            o.elements = args[0]

        for k, v in kwargs.items():
            setattr(o, k, v)
        return o


class MetaContainer(System):
    """Meta class of containers

    A container is a algebric system with elements of some type
    and operators acting on the elements

    Example:
        ```
        from collections import UserString
        class C(metaclass=MetaContainer):
            # container of strings
            element_class = UserString
            # element_name = string

        c = C(strings=[UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        print(c.element_class)
        print(C.element_name)
        print(c.strings)
        print(c.lasting)
        print(c.n_strings)
        print(c[1])
        for a in c:
            print(a)

        c.regester('upper')
        print(c.upper())

        # <class 'collections.UserString'>
        # <property object at 0x1065715e0>
        # ['I', 'love', 'you']
        # for ever
        # 3
        # love
        # I
        # love
        # you
        # ['I', 'LOVE', 'YOU']
        ```
    """

    def __new__(cls, name, bases, attrs):
        # element_class = attrs['element_class']
        if 'element_class' in attrs:
            element_class = attrs['element_class']
        else:
            for base in bases:
                if hasattr(base, 'element_class'):
                    element_class = base.element_class
                    break
            else:
                raise Exception('Have not provided element class yet.')
        if 'element_name' in attrs:
            element_name = attrs['element_name'] + 's'
        else:
            if isinstance(element_class, tuple):
                element_name = get_stem(element_class[0].__name__) + 's'
            else:
                element_name = get_stem(element_class.__name__) + 's'
        setattr(cls, 'element_name', element_name)
        if 'alias' in attrs:
            attrs['alias'].update({element_name:'elements'})
        else:
            attrs['alias'] = {element_name:'elements'}

        return super().__new__(cls, name, bases, attrs)

    def mixin(self, *others):
        class cls(*others, self):
            pass
        return cls


    def __call__(self, *args, **kwargs):
        o = super().__call__(*args, **kwargs)

        if '_environment' in globals():
            o.environment = globals()['_environment']
        return o

    def __getitem__(self, class_):
        return self.set(element_class=class_)

    def __floordiv__(self, n):
        return self.set(default_size=n)


class MetaList(MetaContainer):
    # a list is a container of elements with the same type
    def __new__(cls, name, bases, attrs):
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if isinstance(element_class, tuple):
                raise TypeError('`element_class` should not be a tuple!')
        return super().__new__(cls, name, bases, attrs)

    def __getitem__(self, class_):
        self.element_class = class_
        if isinstance(class_, tuple):
            raise TypeError('`element_class` should not be a tuple!')
        return self


class MetaTuple(MetaContainer):
    # a tuple is a container of elements with different types
    def __new__(cls, name, bases, attrs):
        # constructor of MetaMultiContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if not isinstance(element_class, tuple):
                raise TypeError('`element_class` should be a tuple!')
        return super().__new__(cls, name, bases, attrs)

    def __mul__(self, n):
        raise DeprecationWarning('It is meaningless to multiply the class by a number')


class MetaHighContainer(MetaContainer):
    # High order container is a container of  containers.
    def __new__(cls, name, bases, attrs):
        # constructor of MetaHighContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if not isinstance(element_class, MetaContainer) or isinstance(element_class, tuple) and isinstance(element_class[0], MetaContainer):
                raise TypeError('`element_class` should be an instance of MetaContainer, or a list of such instances')

        def _flatten(self, type_):
            elms = []
            for elm in self._elements:
                elm.extend(elm._elements)
            return elms

        attrs['flatten'] = _flatten

        return super().__new__(cls, name, bases, attrs)


import numpy as np
class MetaArray(ParamType):
    def __new__(cls, name, bases, attrs):
        if 'element_class' in attrs:
            element_class = attrs['element_class']
        else:
            for base in bases:
                if hasattr(base, 'element_class'):
                    element_class = base.element_class
                    break
            else:
                raise Exception('Have not provided element class yet.')
        if not element_class.__name__.startswith('Base') and not issubclass(element_class, (int, float, np.int_, np.float_, np.bool_)):
            raise TypeError('The types of elements should be numbers, i.e. subclass of int or float')
        # if np.ndarray not in bases:
        #     bases = (np.ndarray,) + bases
            # raise Exception('The class should be a subclass of numpy.ndarray!')

        return super().__new__(cls, name, bases, attrs)


    # def __call__(self, *args, **kwargs):
    #     o = super().__call__([], dtype=self.element_class)
    #     for k, v in kwargs.items():
    #         setattr(o, k, v)
    #     if args:
    #         o = super(MetaArray, self).__new__(args, dtype=self.element_class)
    #     return o


if __name__ == '__main__':
    from collections import UserString

    class C(metaclass=MetaContainer):
        element_class = UserString
        alias = {'n_strings': 'n_elements'}
        # element_name = 'string'

    c = C([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
    C.set_methods(n_elems=lambda c: 0)
    print(c.element_class)
    print(C.element_name)
    print(c.strings)
    print(c.lasting)
    print(c.elements, 'one of them is', c.elements[0])
    print(c.n_elements == c.n_strings)

    c.regester_map('upper')
    print(list(c.upper()))
    def n_vowels(s):
        return len([o for o in s if str(o) in 'ieaouIEAOU'])
    c.regester_map('length', n_vowels)
    print(list(c.length()))
    print(c.n_elems())

