#!/usr/bin/env python3

"""
Metaclasses
"""

import copy
from types import MethodType
from collections.abc import Iterable
from operator import attrgetter, methodcaller


def inherit(attrs, attr, bases):
    """Inherit attribute the attributes `attr` from the base classes `bases`

    Args:
        attrs (dict): The attribution dictionary of an object 
        attr (string): An attribution whose value is a dict
        bases (tuple): The base classes

    Returns:
        dict: The updated dictionary of attributions
    """

    v = {}
    for b in bases[::-1]:
        if hasattr(b, attr):
            v.update(getattr(b, attr))

    v.update(attrs.get(attr,  {}))
    # attributes in `v` should not cover the attributes of the object
    v = {k:vk for k, vk in v.items() if k not in attrs}
    attrs[attr] = v
    return attrs


class ParamType(type):
    """Just a wrapper of `type`

    The key-value pairs in `params` could be inherited from super classes, like attributes.
    It make users set and manage parameters of classes or instances more conveniently.

    Example:
        class C(metaclass=ParamType):
            alias = {"a": "A"}
            params = {"A": 1}

        c = C()
        assert c.a == c.A == 1

        class D(C):
            pass
        
        d = D()
        assert d.a == d.A == 1
    """

    def __new__(cls, name, bases=(), attrs={}):

        # inherit alias instead of overwriting it, when setting `alias` for a subclass
        # `alias` is not recommended to use.
        attrs = inherit(attrs, 'alias', bases)

        # inherit params instead of overwriting it, when setting `params` for a subclass
        attrs = inherit(attrs, 'params', bases)

        def _getattr(self, key):
            if key in self.params:
                return self.params[key]
            elif key in self.alias:
                return getattr(self, self.alias[key])
            elif key == 'lambda' and 'lambda_' in self.params:
                return self.params['lambda_']
            else:
                raise AttributeError(f"""`{key}` is neither an attribute of the object of `{self.__class__}`, nor in `params` or `alias`;
                    If you are sure that `{key}` has been defined as an attribute, then you should check the definition statement.
                    If there is no syntax problem, then it is probably about the type error in the statement.""")
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
        # set the attributes dynamically
        for k in args:
            if k in globals():
                setattr(self, k, globals()[k])
            else:
                raise NameError(f"name '{k}' is not defined.")
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def set_params(self, **kwargs):
        # set `params` dynamically
        self.params.update(kwargs)
        return self

    def mixin(self, bases):
        """mixin other base classes

        Args:
            bases (tuple): the base classes
        """

        if isinstance(bases, tuple):
            self.__bases__ = bases + self.__bases__
        else:
            self.__bases__ = (bases,) + self.__bases__
        return self

    def __and__(self, other):
        # The syntax sugar for `class cls(self, other)`
        class cls(self, other):
            pass
        return cls

    def __rand__(self, other):
        # The syntax sugar for `class cls(other, self)`
        class cls(other, self):
            pass
        return cls

    def __call__(self, *args, **kwargs):
        # initalize an object with the same `params` of the class
        obj = super().__call__(*args, **kwargs)
        obj.params = copy.deepcopy(self.params)
        return obj

    def __matmul__(self, deco):
        # The syntax sugar for the decorator
        return deco(self)


class MetaContainer(ParamType):
    """Meta class of containers

    A container is a algebric system with elements of some type
    and operators acting on the elements

    Example:

    ```python
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
    ```

    Output:
    ```
    <class 'collections.UserString'>
    <property object at 0x1065715e0>
    ['I', 'love', 'you']
    for ever
    3
    love
    I
    love
    you
    ['I', 'LOVE', 'YOU']
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
            try:
                return self.__elements[k]
            except:
                if isinstance(k, Iterable):
                    return [self[_] for _ in k]
                else:
                    raise TypeError(f'The index must be int/tuple/slice or list-like iterable object! But what you provided is {type(k)}')

        def _setitem(self, k, v):
            # print(DeprecationWarning('get item directly is not recommended now.'))
            self.__elements[k] =v
            self.after_setter()

        def _iter(self):
            return iter(self.__elements)

        def _len(self):
            return len(self.__elements)

        def _contains(self, e):
            return e in self.__elements

        attrs.update(
            {'__getitem__': _getitem,
            '__setitem__': _setitem,
            '__iter__': _iter,
            '__len__': _len,
            '__contains__': _contains}
            )

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
            "apply": _apply
            })

        # def _type_check(self):
        #     return all(isinstance(elm, self.element_class) for elm in self.__elements)

        # attrs['type_check'] = _type_check

        def _getstate(self):
            return {'element_class':self.element_class,
            'default_size':self.default_size,
            'elements':self.elements,
            'params':self.params}

        def _setstate(self, state):
            self.element_class = state.get('element_class', self.__class__.element_class)
            self.default_size = state.get('default_size', self.__class__.default_size)
            self.elements = state.get('elements', [])
            self.params = state.get('params', {})

        attrs.update(
            {'__getstate__': _getstate,
            '__setstate__': _setstate
            })

        def _isa(self, cls):
            return isinstance(self, cls) and self.element_class == cls.element_class

        attrs.update({'isa': _isa})

        """
        Regester maps and operands
        if the mapping f is regestered, then A owns method f, automatically
        f(A) := {f(a), a in A} where f is a method of A.
        """

        def _regester_map(self, name, key=None, force=True):
            if key is None:
                key = methodcaller(name)

            _map = self.map if hasattr(self, 'map') else map

            def m(obj, *args, **kwargs):
                return _map(key, obj.__elements, *args, **kwargs)
                
            if not force and hasattr(self, name):
                raise RegesterError(self.__class__, name)
            setattr(self, name, MethodType(m, self))

        attrs.update({
            'regester_map': _regester_map
        })

        def _regester(self, name, key, force=True):
            if not force and hasattr(self, name):
                raise RegesterError(self.__class__, name)
            setattr(self, name, MethodType(key, self))

        attrs.update({
            'regester': _regester
        })

        return super().__new__(cls, name, bases, attrs)

    def __call__(self, *args, **kwargs):
        o = super().__call__()
        o.params = copy.deepcopy(self.params)

        if args:
            elements = []
            if isinstance(self.element_class, tuple):
                for e, e_c in zip(args[0], self.element_class):
                    if not isinstance(e, e_c):
                        try:
                            e = e.copy(type_=e_c)
                        except:
                            e = e_c(e)
                    elements.append(e)
            else:
                e_c = self.element_class
                for e in args[0]:
                    if not isinstance(e, e_c):
                        try:
                            e = e.copy(type_=e_c)
                        except:
                            e = e_c(e)
                    elements.append(e)

            o.__elements = elements
            # for e in o.__elements:  # consider in future
            #     e.__system = o
        else:
            raise TypeError('missing a list/tuple of elements as the unique positional argument!')

        for k, v in kwargs.items():
            setattr(o, k, v)

        return o

    def __getitem__(self, class_):
        """Helper to construct a container

            `C[a] // n` is equiv. to

            ```
            class C(metaclass=MetaContainer):
                element_class = a
                default_size = n
            ```
        
        Args:
            class_: `element_class` of the container
        
        Returns:
            A container class
        """
        return self.set(element_class=class_)

    def __ifloordiv__(self, n):
        """The syntax sugar for `self.set(default_size=n)`
        
        Keyword Arguments:
            n {number} -- the number assigned to default_size

        Returns:
            The container class
        """

        return self.set(default_size=n)

    def __floordiv__(self, n):
        """The syntax sugar for `self.set(default_size=n)` but return a copy
        
        Keyword Arguments:
            n {number} -- the number assigned to default_size

        Returns:
            A copyed class
        """

        class cls(self):
            default_size = n
        cls._name = self.__name__
        return cls

    def random(self, n_elements=None, *args, **kwargs):
        """Generate a container randomly
        
        Arguments:
            **kwargs -- set size of the container if use the alias of n_elements
        
        Keyword Arguments:
            n_elements {number} -- the number of elements (default: {None})

        Returns:
            The container class
        """

        for k, v in kwargs.items():
            if k in self.alias and self.alias[k] == 'n_elements':
                n_elements = v
                del kwargs[k]
                break
        if isinstance(self.element_class, tuple):
            return self([C.random(*args, **kwargs) for C in self.element_class])
        else:
            n_elements = n_elements or self.default_size
            return self([self.element_class.random(*args, **kwargs) for _ in range(n_elements)])


class System(MetaContainer):
    """Metaclass of systems, considered in future!

    A system is a type of container, that defines operators on them.
    """
    pass
    
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

    def __call__(self, *args, **kwargs):
        o = super().__call__(*args, **kwargs)
        for e in o:  # consider in future
            if not isinstance(e, self.element_class):
                raise TypeError(f'"{e}" is not an instance of type "{self.element_class}"')
        return o


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
        raise DeprecationWarning('It is meaningless to do `//` on the class {self.__name__} with a number.')

    def __call__(self, *args, **kwargs):
        o = super().__call__(*args, **kwargs)
        for e, t in zip(o, self.element_class):  # consider in future
            if not isinstance(e, t):
                raise TypeError(f'"{e}" is not an instance of type "{t}"')
        return o


class MetaHighContainer(MetaContainer):
    # High order container is a container of containers.

    def __new__(cls, name, bases, attrs):
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if (not isinstance(element_class, MetaContainer)
                and isinstance(element_class, tuple) and not any(isinstance(ec, MetaContainer) for ec in element_class)
                and not isinstance(element_class, ParamType)):
                raise TypeError('`element_class` should be an instance of `MetaContainer`, or a tuple where one element is an instance of `MetaContainer`.')

        def _flatten(self, type_):
            from toolz import concat
            return concat(elm.__elements for elm in self.__elements)

        attrs['flatten'] = _flatten

        return super().__new__(cls, name, bases, attrs)


class MetaSingle(MetaContainer):
    # metaclass for the containers that have only one element

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


# import array
# import numpy as np

# def array_check(bases):
#     if array.array in bases or np.ndarray in bases:
#         return True
#     else:
#         for base in bases:
#             if issubclass(base, (array.array, np.ndarray)):
#                 return True
#         else:
#             return False


class MetaArray(ParamType):

    """Metaclass for chromosomes

    Chromosomes could be seen as a container of genes. But we implement them
    by the arrays for convenience.
    """

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
        # if not issubclass(element_class, (int, float, np.int_, np.float_, np.bool_)) or isinstance(element_class, str):
        #     raise TypeError('The types of elements should be a subclass of int or float, or a string representing a type')

        # if not array_check(bases):
        #     raise Exception(f'The class `{name}` should be a subclass of numpy.ndarray or array.array!')

        return super().__new__(cls, name, bases, attrs)

    def __ifloordiv__(self, n):
        """The syntax sugar for `self.set(default_size=n)`
        
        Keyword Arguments:
            n {number} -- the number assigned to default_size

        Returns:
            The container class
        """

        return self.set(default_size=n)

    def __floordiv__(self, n):

        class cls(self):
            default_size = n
        cls._name = self.__name__
        return cls
