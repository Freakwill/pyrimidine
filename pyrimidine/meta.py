#!/usr/bin/env python3


"""
Metaclasses
"""

from types import MethodType
from collections.abc import Iterable
from operator import attrgetter, methodcaller


def inherit(attrs, attr, bases):
    """inherit attribute `attr` from `bases`
    
    Args:
        attrs (dict): THe dictionary of attributions 
        attr (string): An attribution
        bases (tuple): The base classes
    
    Returns:
        dict: The updated dictionary of attributions
    """
    v = {}
    for b in bases:
        if hasattr(b, attr) and hasattr(b, attr):
            v.update(getattr(b, attr))

    v.update(attrs.get(attr,  {}))
    v = {k:vk for k, vk in v.items() if k not in attrs}
    attrs[attr] = v
    return attrs


class ParamType(type):
    """just a wrapper of `type`

    then key-value pairs in `params` could be inherited from super classes, like attributes.
    It make users set and manage parameters of classes or instances more conveniently.
    """

    def __new__(cls, name, bases=(), attrs={}):
        # inherit alias instead of overwriting it, when setting `alias` for a subclass
        # alias is not recommended to use!
        attrs = inherit(attrs, 'alias', bases)

        # inherit params instead of overwriting it, when setting `params` for a subclass
        attrs = inherit(attrs, 'params', bases)

        def _getattr(self, key):
            if key in self.__dict__:
                return self.__dict__[key]
            elif key in self.params:
                return self.params[key]
            elif key in self.alias:
                return getattr(self, self.alias[key])
            else:
                raise AttributeError(f'`{key}` is neither an attribute of the object of `{self.__class__}`, nor in `params` or `alias`')
        attrs['__getattr__'] = _getattr

        def _setattr(self, key, value):
            if key in self.__dict__:
                self.__dict__[key] = value
            elif key in self.params:
                self.params[key] = value
            elif key in self.alias:
                setattr(self, self.alias[key], value)
            else:
                object.__setattr__(self, key, value)
        attrs['__setattr__'] = _setattr

        # if 'check' in attrs:
        #     attrs['check'](cls)

        return super().__new__(cls, name, bases, attrs)


    @classmethod
    def __prepare__(cls, name, bases):
        return {"alias":{}, "params":{}}


    def set(self, *args, **kwargs):
        for k in args:
            setattr(self, k, globals()[k])
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


    def set_method(self, *args, **kwargs):
        for k in args:
            setattr(self, k, MethodType(globals()[k], self))
        for k, m in kwargs.items():
            setattr(self, k, MethodType(m, self))
        return self


    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self


    def mixin(self, bases):
        if isinstance(bases, tuple):
            self.__bases__ = bases + self.__bases__
        else:
            self.__bases__ = (bases,) + self.__bases__
        return self


    def __and__(self, other):
        class cls(self, other):
            pass
        return cls


class MetaContainer(ParamType):
    """Meta class of containers

    A container is a algebric system with elements of some type
    and operators acting on the elements

    Example:
        ```
        from collections import UserString
        class C(metaclass=MetaContainer):
            # container of strings
            element_class = UserString
            alias = {'strings': 'elements'}

        c = C(strings=[UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        print(c.element_class)
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
        """
        Users have to define `element_class` in the class.
        """
        if 'element_class' in attrs:
            element_class = attrs['element_class']
        else:
            for base in bases:
                if hasattr(base, 'element_class'):
                    element_class = base.element_class
                    break
            else:
                raise Exception('Have not provided element class yet.')
        
        def _getitem(self, k):
            # print(DeprecationWarning('get item directly is not recommended now.'))
            return self.__elements[k]

        def _setitem(self, k, v):
            # print(DeprecationWarning('get item directly is not recommended now.'))
            self.__elements[k] =v

        def _iter(self):
            return iter(self.__elements)

        def _len(self):
            # if hasattr(self, '__n_elements'):
            #     return getattr(self, '__n_elements')
            return len(self.__elements)

        def _contains(self, e):
            # if hasattr(self, '__n_elements'):
            #     return getattr(self, '__n_elements')
            return e in self.__elements

        attrs.update(
            {'__getitem__': _getitem,
            '__setitem__': _setitem,
            '__iter__': _iter,
            '__len__': _len,
            '__contains__': _contains})

        def _get_all(self, attr_name):
            return map(attrgetter(attr_name), self.__elements)

        def _apply(self, f, *args, **kwargs):
            return self.map(lambda o: f(o, *args, **kwargs), self.__elements)

        @property
        def _n_elements(self):
            return len(self.__elements)

        @property
        def _elements(self):
            return self.__elements

        @_elements.setter
        def _elements(self, x):
            self.__elements = x
            if hasattr(self, 'after_setter'):
                self.after_setter()

        attrs.update(
            {"elements": _elements,
            "n_elements": _n_elements,
            "get_all": _get_all,
            "apply": _apply
            })

        def _type_check(self):
            return all(isinstance(elm, self.element_class) for elm in self.__elements)

        attrs['type_check'] = _type_check

        def _getstate(self):
            return {'element_class':self.element_class, 'default_size':self.default_size,
            'elements':self.elements, 'params':self.params}

        def _setstate(self, state):
            self.element_class = state.get('element_class', self.__class__.element_class)
            self.default_size = state.get('default_size', self.__class__.default_size)
            self.elements = state.get('elements', [])
            self.params = state.get('params', {})

        attrs.update(
            {'__getstate__': _getstate,
            '__setstate__': _setstate
            })

        """
        Regester maps and operands
        if the mapping f is regestered, then A owns method f, automatically
        f(A) := {f(a), a in A} where f is a method of A.
        """
        def _regester_map(self, name, key=None, force=True):
            if key is None:
                key = methodcaller(name)
            def m(obj, *args, **kwargs):
                return obj.map(key, obj.__elements, *args, **kwargs)
            if not force and hasattr(self, name):
                raise AttributeError(f'`{name}` is an attribute of {self.__class__.__name__}, and would not be regestered.')
            setattr(self, name, MethodType(m, self))

        attrs.update({
            'regester_map': _regester_map
        })

        return super().__new__(cls, name, bases, attrs)


    def __call__(self, *args, **kwargs):
        o = super().__call__()

        if args:
            o.__elements = args[0]
            # for e in o.__elements:  # consider in future
            #     e.__system = o
        else:
            raise TypeError('missing a list/tuple of elements as the unique positional argument!')

        for k, v in kwargs.items():
            setattr(o, k, v)

        # if '_environment' in globals():
        #     o.environment = globals()['_environment']
        return o


    def __getitem__(self, class_):
        return self.set(element_class=class_)


    def __floordiv__(self, n):
        return self.set(default_size=n)


# class System(MetaContainer):
#     """Metaclass of systems, considered in future!

#     A system is a type of container, that defines operators on them.
#     """
    
#     def __new__(cls, name, bases=(), attrs={}): 
#         """
#         Regester maps and operands
#         if the mapping f is regestered, then A owns method f, automatically
#         f(A) := {f(a), a in A} where f is a method of A.
#         """

#         def _regester_operator(self, name, key=None):
#             if hasattr(self, name):
#                 raise AttributeError(f'`{name}` is an attribute of {self.__class__.__name__}, and would not be regestered.')
#             self.__operators.append(key)
#             setattr(self, name, MethodType(key, self))

#         def _element_regester(self, name, e):
#             if hasattr(self, e):
#                 raise AttributeError(f'`{e}` is an attribute of {self.__class__.__name__}, would not be regestered.')
#             self.__elements.append(e)
#             setattr(self, name, e)

#         attrs.update({
#             '__operators': []
#             'regester_operator': _regester_operator,
#             'regester_element': _regester_element
#         })

#         return super().__new__(cls, name, bases, attrs)


#     def __call__(self, *args, **kwargs):
#         o = super().__call__(*args, **kwargs)

#         for e in o.__elements:  # consider in future
#             e.__system = o
#         return o



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


    def __floordiv__(self, n):
        raise DeprecationWarning('It is meaningless to do `//` on the class with a number.')


class MetaHighContainer(MetaContainer):
    # High order container is a container of containers.
    def __new__(cls, name, bases, attrs):
        # constructor of MetaHighContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if (not isinstance(element_class, MetaContainer)
                and isinstance(element_class, tuple) and not isinstance(element_class[0], MetaContainer)
                and not isinstance(element_class, ParamType)):
                raise TypeError('`element_class` should be an instance of MetaContainer, or a list of such instances.')

        def _flatten(self, type_):
            elms = []
            for elm in self.__elements:
                elm.extend(elm.__elements)
            return elms

        attrs['flatten'] = _flatten

        return super().__new__(cls, name, bases, attrs)


class MetaSingle(MetaContainer):
    def __new__(cls, name, bases, attrs):
        if 'n_elements' in attrs and attrs['n_elements'] !=1:
            raise ValueError('n_elements should be 1!')
        else:
            attrs['n_elements'] = 1
        return super().__new__(cls, name, bases, attrs)

    def __call__(self, *args, **kwargs):
        o = super().__call__(*args, **kwargs)

        if o.n_elements != 1:
            raise ValueError('There should be only 1 element!')
        return o


import numpy as np

# import array

# def array_check(bases):
#     if array.array in bases or np.ndarray in bases:
#         return True
#     else:
#         for base in bases:
#             if isinstance(base, (array.array, np.ndarray)):
#                 return True
#         else:
#             return False


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

        # if not name.startswith('Base') and array_check(bases):
        #     raise Exception(f'The class `{name}` should be a subclass of numpy.ndarray or array.array!')

        return super().__new__(cls, name, bases, attrs)

    def __floordiv__(self, n):
        return self.set(default_size=n)

