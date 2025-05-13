from quompiler.circuits.quimb_factory.qumb_factory import QuimbFactory

from tests.qompiler.mock_fixtures import mock_factory_manager


def test_create_factory(mocker):
    man = mock_factory_manager(target="QUIMB")
    factory = man.create_factory()
    assert factory is not None
    assert isinstance(factory, QuimbFactory)
