from quompiler.utils.mgen import random_UnitaryM_2l
from tests.qompiler.mock_fixtures import mock_factory_manager

man = mock_factory_manager(emit="SINGLET", ancilla_offset=100, target="QUIMB")


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)
    factory = man.create_factory()
    quimbC = factory.get_builder()
    assert quimbC is None
