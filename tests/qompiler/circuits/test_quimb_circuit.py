from quompiler.circuits.qfactory import QFactory
from quompiler.config.config_manager import ConfigManager
from quompiler.construct.types import QompilePlatform
from quompiler.utils.mgen import random_UnitaryM_2l

override = dict(emit="SINGLET", ancilla_offset=100, target="QUIMB")
config = ConfigManager().merge(override).create_config()


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)
    factory = QFactory(config)
    quimbC = factory.get_builder(QompilePlatform.QUIMB)
    assert quimbC is not None
