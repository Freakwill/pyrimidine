#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import MethodType, Iterable


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


class MetaContainer(type):
    """Container meta class

    A container is a algebric system with elements of some type
    and operators acting on the elements

    Example:
        ```
        from collections import UserString
        class C(metaclass=MetaContainer):
            element_class = UserString
            # element_name = string

        c = C(strings=[UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        print(c.element_class)
        print(C.strings)
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
    
    Extends:
        type
    """
    def __new__(cls, name, bases, attrs):
        # element_class = attrs['element_class']
        if 'element_class' in attrs:
            element_class = attrs['element_class']
        else:
            raise Exception('Provide element class.')
        if 'element_name' in attrs:
            element_name = attrs['element_name'] + 's'
        else:
            element_name = get_stem(element_class.__name__) + 's'

        element_name_ = '__' + element_name

        def _iter(self):
            return iter(getattr(self, element_name_))

        def _getitem(self, k):
            return getattr(self, element_name_)[k]

        def _len(self, k):
            if hasattr(self, 'n_' + element_name):
                return getattr(self, 'n_' + element_name)
            return len(getattr(self, element_name_))

        attrs['__getitem__'] = _getitem
        attrs['__len__'] = _len
        attrs['__iter__'] = _iter

        @property
        def _elements(self):
            return getattr(self, element_name_)

        @_elements.setter
        def _elements(self, x):
            setattr(self, element_name_, x)
            setattr(self, 'n_' + element_name, len(x))
        attrs[element_name] = _elements
        attrs['elements'] = _elements

        # if 'random' not in attrs:
        #     print(Warning('Define a `random` method'))
        def regester(self, m):
            if hasattr(self, m):
                raise AttributeError(f'`{m}` is an attribute of {self.__class__.__name__}, would not be regestered.')
            def _m(obj, *args, **kwargs):
                return [getattr(a, m)(*args, **kwargs) for a in obj]
            setattr(self, m, MethodType(_m, self))
        attrs['regester'] = regester

        return type.__new__(cls, name, bases, attrs)

    def __call__(self, *args, **kwargs):
        o = super(MetaContainer, self).__call__()
        for k, v in kwargs.items():
            setattr(o, k, v)
        return o

class MetaHighContainer(MetaContainer):
    def __new__(cls, name, bases, attrs):
        # constructor of MetaHighContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if not isinstance(element_class, Iterable):

        else:
            raise Exception('Provide element class.')
        obj = super(MetaHighContainer, cls).__new__(cls, name, bases, attrs)
        return obj



if __name__ == '__main__':
    from collections import UserString
    class C(metaclass=MetaContainer):
        element_class = UserString
        # element_name = string

    c = C(strings=[UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
    print(c.element_class)
    print(C.strings)
    print(c.strings)
    print(c.lasting)
    print(c.n_strings)
    print(c[1])
    for a in c:
        print(a)

    c.regester('upper')
    print(c.upper())


