#!/usr/bin/env python3


class UnknownSizeError(Exception):
    def __init__(self, cls):
        self.cls = cls
        
    def __str__(self):
        return f'The size of `{self.cls}` is unkown, the object could not be generated.'

class UnavalibleAttributeError(Exception):
    def __init__(self, cls, attr_name):
        self.cls = cls
        self.attr_name = attr_name
        
    def __str__(self):
        return f'Did not define attribute `{self.attr_name}` for `{self.cls}`.'