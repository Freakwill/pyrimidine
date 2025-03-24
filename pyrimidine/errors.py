#!/usr/bin/env python3

"""
Custom exception classes to handle specific error scenarios,
during the object generation and attribute access processes.
"""

class UnknownSizeError(Exception):
    # raise the error, if the size of class is unknown.

    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return f'The size of the class `{self.cls.__name__}` is unknown, the object could not be generated.'


class UnavalibleAttributeError(AttributeError):
    # raise the error, if some special attribute is undefined.

    def __init__(self, cls, attr_name):
        self.cls = cls
        self.attr_name = attr_name

    def __str__(self):
        return f'Did not define the attribute `{self.attr_name}` for the class `{self.cls.__name__}`.'


class RegesterError(AttributeError):
    # raise the error, if try to redefine some special attribute.

    def __init__(self, cls, attr_name):
        self.cls = cls
        self.attr_name = attr_name

    def __str__(self):
        return AttributeError(f'`{attr_name}` is an attribute of the class `{self.cls.__name__}`, and would not be redefined.')
