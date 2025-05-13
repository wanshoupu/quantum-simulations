from quompiler.circuits.quimb_factory.quimb_builder import QuimbBuilder
from quompiler.utils.mgen import random_UnitaryM_2l
from tests.qompiler.qompile.mock_fixtures import mock_config


def test_create_builder(mocker):
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)

    config = mock_config(mocker, emit="SINGLET", ancilla_offset=100)
    quimbC = QuimbBuilder(config.device)
    quimbC.build_gate(array)
