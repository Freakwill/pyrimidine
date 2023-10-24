#!/usr/bin/env python3


"""
Metaclasses
"""

from types import MethodType
from collections.abc import Iterable
from operator import attrgetter



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

    Define `params` in classes whose metaclass of ParamType,
    then key-value pairs in `params` could be inherited from super classes, like attributes.
    It make users set and manage parameters of classes or instances more easily.
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
                raise AttributeError(f'`{key}` is neither an attribute of the object, nor in `param` or `alias`')
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

        if 'check' in attrs:
            attrs('check')(cls)

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

    def set_methods(self, *args, **kwargs):
        for k in args:
            setattr(self, k, globals()[k])
        for k, m in kwargs.items():
            setattr(self, k, MethodType(m, self))
        return self


class System(ParamType):
    """Metaclass of systems

    A system consists of a set of elements and operators acting on them

    It is refered to an algebraic system.
    """
    
    def __new__(cls, name, bases=(), attrs={}):
        # Create with regesters
        
        def _iter(self):
            return iter(self.__elements)

        def _getitem(self, k):
            # print(DeprecationWarning('get item directly is not recommended now.'))
            return self.__elements[k]

        def _len(self):
            if hasattr(self, '__n_elements'):
                return getattr(self, '__n_elements')
            return len(self.__elements)

        attrs.update(
            {'__getitem__': _getitem,
            '__len__': _len,
            '__iter__': _iter})

        def _get_all(self, attr_name):
            return map(attrgetter(attr_name), self.__elements)


        def _apply(self, f, *args, **kwargs):
            return map(lambda o: f(o, *args, **kwargs, self.__elements))

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

        """
        Regester maps and operands
        if the mapping f is regestered, then A owns method f, automatically
        f(A) := {f(a), a in A} where f is a method of A.
        """
        def _regester_map(self, name, key=None, force=True):
            if key is None:
                key = lambda e: getattr(e, name)()
            def m(obj):
                return map(key, obj.elements)
            if not force and hasattr(self, name):
                raise AttributeError(f'`{name}` is an attribute of {self.__class__.__name__}, and would not be regestered.')
            setattr(self, name, MethodType(m, self))


        def _regester_op(self, name, key=None, force=True):
            if key is None:
                key = lambda e, o: getattr(e, name)(o)
            def m(obj):
                return map(key, zip(obj.elements, other.elements))
            if not force and hasattr(self, name):
                raise AttributeError(f'`{name}` is an attribute of {self.__class__.__name__}, and would not be regestered.')
            setattr(self, name, MethodType(m, self))

        # attrs = {
        #     'regester_op': _regester_op,
        #     'regester_map': _regester_map
        # }
        
        # def _operator_regester(self, m):
        #     if hasattr(self, m):
        #         raise AttributeError(f'`{m}` is an attribute of {self.__class__.__name__}, would not be regestered.')
        #     def _m(obj, *args, **kwargs):
        #         return [getattr(a, m)(*args, **kwargs) for a in obj]
        #     setattr(self, m, MethodType(_m, self))

        # def _element_regester(self, e):
        #     if hasattr(self, e):
        #         raise AttributeError(f'`{e}` is an attribute of {self.__class__.__name__}, would not be regestered.')
        #     @property
        #     def _p(obj):
        #         return [getattr(a, e) for a in obj]
        #     setattr(self, e, _p)

        # attrs.update({'operator_regester': _operator_regester,
        # 'element_regester': _element_regester})

        return super().__new__(cls, name, bases, attrs)


    def __call__(self, *args, **kwargs):
        o = super().__call__()

        if args:
            o.__elements = args[0]
        # else:
        #     raise Exception('Have not provided a list of elements as the unique positional argument!')

        for k, v in kwargs.items():
            setattr(o, k, v)

        if '_environment' in globals():
            o.environment = globals()['_environment']
        return o


class MetaContainer(System):
    """Meta class of containers

    A container is a algebric system with elements of some type
    and operators acting on the elements

    Example:
        ```
        from collections import UserString
        class C(metaclass=MetaContainer):
            # container of strings
            element_class = UserString
            # element_name = string

        c = C(strings=[UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
        print(c.element_class)
        print(C.element_name)
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

        if 'element_class' in attrs:
            element_class = attrs['element_class']
        else:
            for base in bases:
                if hasattr(base, 'element_class'):
                    element_class = base.element_class
                    break
            else:
                raise Exception('Have not provided element class yet.')
        if 'element_name' in attrs:
            element_name_ = attrs['element_name'] + 's'
        else:
            if isinstance(element_class, tuple):
                element_name_ = get_stem(element_class[0].__name__) + 's'
            else:
                element_name_ = get_stem(element_class.__name__) + 's'

        if element_name_ not in attrs['alias']:
            attrs['alias'].update({element_name_: 'elements'})

        for k, v in attr['alias'].items():
            if v == 'elements' and 'n_' + k not in attr['alias']:
                attrs['alias'].update({'n_' + k: 'n_elements'})

        return super().__new__(cls, name, bases, attrs)


    def __getitem__(self, class_):
        return self.set(element_class=class_)

    def __floordiv__(self, n):
        return self.set(default_size=n)


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
        raise DeprecationWarning('It is meaningless to do floordiv on the class by a number')


class MetaHighContainer(MetaContainer):
    # High order container is a container of containers.
    def __new__(cls, name, bases, attrs):
        # constructor of MetaHighContainer
        if 'element_class' in attrs:
            element_class = attrs['element_class']
            if (not isinstance(element_class, MetaContainer)
                and isinstance(element_class, tuple) and not isinstance(element_class[0], MetaContainer)
                and not isinstance(element_class, ParamType)):
                raise TypeError('`element_class` should be an instance of MetaContainer, or a list of such instances')

        def _flatten(self, type_):
            elms = []
            for elm in self.__elements:
                elm.extend(elm.__elements)
            return elms

        attrs['flatten'] = _flatten

        return super().__new__(cls, name, bases, attrs)


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



if __name__ == '__main__':
    from collections import UserString

    class C(metaclass=MetaContainer):
        element_class = UserString
        # alias = {'n_strings': 'n_elements'}
        # element_name = 'string'

        def foo(self):
            pass

        def after_setter(self):
            self.fitness = 2

    class D(C):
        pass

    c = C([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
    C.set_methods(n_elems=lambda c: 1)
    print(c.element_class)
    print(C.element_name)
    print(c.strings)
    print(c.lasting)
    print(c.elements, 'one of them is', c.elements[0])
    print(c.n_elements == c.n_strings)

    c.regester_map('upper')
    print(list(c.upper()))
    def n_vowels(s):
        return len([o for o in s if str(o) in 'ieaouIEAOU'])
    c.regester_map('length', n_vowels)
    print(list(c.length()))
    print(c.n_elems())

    c = D([UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
    c.fitness = 1
    c.strings += [UserString('wow')]
    print(c.elements)
    print(c.fitness)

