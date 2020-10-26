import pytest
from pyrimidine import *

class Test_Pyrimidine:
    def test_get_stem(self):
        assert get_stem('ILoveYou') == 'you'
    
    def test_meta(self):
        from collections import UserString
        class C(metaclass=MetaContainer):
            element_class = UserString
            # element_name = string

            c = C(strings=[UserString('I'), UserString('love'), UserString('you')], lasting='for ever')
            assert hasattr(c, 'strings')
            assert len(c) == c.n_elements
            for a in c:
                assert isinstance(a, UserString)

            c.regester_map('upper')
            assert hasattr(c, 'upper')

    def test_sga(self):
        try:
            _Population = SGAPopulation[BinaryIndividual] // 20
        except Exception as e:
            raise e
