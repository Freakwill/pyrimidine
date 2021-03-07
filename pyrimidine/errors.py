#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class UnknownSizeError(Exception):
    def __init__(self, cls):
        self.cls = cls
        
    def __str__(self):
        return f'The size of {self.cls} is unkown, the object could not be generated.'