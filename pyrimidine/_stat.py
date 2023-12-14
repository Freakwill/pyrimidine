#!/usr/bin/env python3

from typing import Iterable


def _call(s, obj):
    """Core function of `ezstat`

    A functional extension for s(obj) or obj.s or obj.s()
    If s is an string, then it only returns obj.s or obj.s().

    s could be a tuple or dict, as a lifted form, though it is not recommended.
    
    Arguments:
        s {function | string} -- Statistics
        obj -- object of statistics
    
    Return:
        a number or a tuple of numbers, as the value of statistics
    """
    if isinstance(s, str):
        if not hasattr(obj, s):
            raise ValueError(f"the object '{obj}' of '{obj.__class__}' has no attribute '{s}'")
        f = getattr(obj, s)
        r = f() if callable(f) else f
    elif callable(s):
        r = s(obj)
    elif isinstance(s, (int, float)):
        print(Warning('Deprecated to use a constant number!'))
        r = s
    elif isinstance(s, tuple):
        return tuple(_call(si, obj) for si in s)
    elif isinstance(s, dict):
        return {i:_call(si, obj) for i, si in s.items()}
    else:
        raise TypeError(f"The type of `{s}` is not permissible!") 

    return r


class Statistics(dict):
    """
    Statistics is a type of dict{str:function},
    where `function` will act on the object of statistics.

    As the value of dict, `function` has not to be a function.
    If it is a string, then the attribute of object will be called.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        for k, _ in obj.items():
            if not isinstance(k, str):
                raise TypeError(f'The keys must be strings, but `{k}` is not a string.')
        return obj

    def __call__(self, obj, split:bool=False):

        res = {}
        for k, s in self.items():
            if s is True:
                s = k.lower().replace(' ', '_').replace('-', '_')
            r = _call(s, obj)
            if split and isinstance(r, Iterable):
                res.update({f"{k}[{i}]": ri for i, ri in enumerate(r)})
            elif isinstance(r, dict):
                res.update({f"{k}[{i}]": ri for i, ri in r.items()})
            else:
                res[k] = r
        return res

