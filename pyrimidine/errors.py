#!/usr/bin/env python3


class UnknownSizeError(Exception):

    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return f'The size of class `{self.cls.__name__}` is unkown, the object could not be generated.'


class UnavalibleAttributeError(AttributeError):

    def __init__(self, cls, attr_name):
        self.cls = cls
        self.attr_name = attr_name

    def __str__(self):
        return f'Did not define attribute `{self.attr_name}` for the class `{self.cls.__name__}`.'


class RegesterError(AttributeError):

    def __init__(self, cls, attr_name):
        self.cls = cls
        self.attr_name = attr_name

    def __str__(self):
        return AttributeError(f'`{attr_name}` is an attribute of the class `{self.cls.__name__}`, and would not be redefined.')
