from quompiler.circuits.create_factory import create_factory
from quompiler.circuits.qiskit_factory.qiskit_builder import QiskitBuilder
from quompiler.utils.mgen import random_UnitaryM_2l
from tests.qompiler.qompile.mock_fixtures import mock_config


def test_create_builder(mocker):
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)

    config = mock_config(mocker, emit="SINGLET", ancilla_offset=100)
    config.target = 'QISKIT'
    factory = create_factory(config)
    qiskitC = factory.get_qompiler()
    assert qiskitC is None
