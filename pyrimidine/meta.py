#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


class ContainerMetaClass(type):
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

        def _getitem(self, k):
            return getattr(self, element_name_)[k]

        def _len(self, k):
            if hasattr(self, 'n_' + element_name):
                return getattr(self, 'n_' + element_name)
            return len(getattr(self, element_name_))

        attrs['__getitem__'] = _getitem
        attrs['__len__'] = _len

        @property
        def _elements(self):
            return getattr(self, element_name_)
        attrs[element_name] = _elements

        @_elements.setter
        def _elements(self, x):
            setattr(self, element_name_, x)
            setattr(self, 'n_' + element_name, len(x))
        attrs[element_name] = _elements

        return type.__new__(cls, name, bases, attrs)

    def __call__(self, **x):
        o = super(ContainerMetaClass, self).__call__()
        for k, v in x.items():
            setattr(o, k, v)
        return o
