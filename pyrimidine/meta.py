#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import MethodType
from collections.abc import Iterable


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
                raise Exception('Provide element class.')
        if 'element_name' in attrs:
            element_name = attrs['element_name'] + 's'
        else:
            if isinstance(element_class, tuple):
                element_name = get_stem(element_class[0].__name__) + 's'
            else:
                element_name = get_stem(element_class.__name__) + 's'
        cls.element_name = element_name

        def _iter(self):
            return iter(self.__elements)

        def _getitem(self, k):
            return self.__elements[k]

        def _len(self):
            if hasattr(self, 'n_' + element_name):
                return getattr(self, 'n_' + element_name)
            return len(self.__elements)

        attrs['__getitem__'] = _getitem
        attrs['__len__'] = _len
        attrs['__iter__'] = _iter

        @property
        def _elements(self):
            return self.__elements

        @_elements.setter
        def _elements(self, x):
            self.__elements = x
            L = len(x)
            self.n_elements = len(x)
            setattr(self, 'n_' + element_name, len(x))
        attrs['elements'] = attrs[element_name] = _elements

        # if 'random' not in attrs:
        #     print(Warning('Define a `random` method'))
        def _regester(self, m):
            if hasattr(self, m):
                raise AttributeError(f'`{m}` is an attribute of {self.__class__.__name__}, would not be regestered.')
            def _m(obj, *args, **kwargs):
                return [getattr(a, m)(*args, **kwargs) for a in obj]
            setattr(self, m, MethodType(_m, self))
        attrs['regester'] = _regester


        def _type_check(self):
            return all(isinstance(elm, self.element_class) for elm in self)
        attrs['type_check'] = _type_check


        params = attrs.get('params', {})
        for b in bases:
            if hasattr(b, 'params') and b.params:
                params.update(b.params)
        attrs['params'] = params

        return type.__new__(cls, name, bases, attrs)

    def __call__(self, *args, **kwargs):
        if args:
            self.__elements, = args
        o = super(MetaContainer, self).__call__(*args, **kwargs)
        for k, v in kwargs.items():
            setattr(o, k, v)
        return o

    def __getitem__(self, class_):
        self.element_class = class_
        return self

class MetaMonoContainer(MetaContainer):
    def __new__(cls, name, bases, attrs):
        # constructor of MetaMultiContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if isinstance(element_class, tuple):
                raise TypeError('`element_class` should not be a tuple!')
        return super(MetaMultiContainer, cls).__new__(cls, name, bases, attrs)


class MetaMultiContainer(MetaContainer):
    def __new__(cls, name, bases, attrs):
        # constructor of MetaMultiContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if not isinstance(element_class, tuple):
                raise TypeError('`element_class` should be a tuple!')
        return super(MetaMultiContainer, cls).__new__(cls, name, bases, attrs)

class MetaHighContainer(MetaContainer):
    def __new__(cls, name, bases, attrs):
        # constructor of MetaHighContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if not isinstance(element_class, MetaContainer) or isinstance(element_class, tuple) and isinstance(element_class[0], MetaContainer):
                raise TypeError('`element_class` should be an instance of MetaContainer, or a list of such instances')
        return super(MetaHighContainer, cls).__new__(cls, name, bases, attrs)



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


