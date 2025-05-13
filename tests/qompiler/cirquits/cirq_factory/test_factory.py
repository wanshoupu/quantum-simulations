from quompiler.circuits.create_factory import create_factory
from quompiler.circuits.quimb_factory.qumb_factory import QuimbFactory
from tests.qompiler.qompile.mock_fixtures import mock_config


def test_create_factory(mocker):
    config = mock_config(mocker, emit="SINGLET", ancilla_offset=100)
    config.target = 'QUIMB'
    factory = create_factory(config)
    assert factory is not None
    assert isinstance(factory, QuimbFactory)
