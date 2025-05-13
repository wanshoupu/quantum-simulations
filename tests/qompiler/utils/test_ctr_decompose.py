import random

from quompiler.circuits.factory_manager import FactoryManager
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_CtrlGate, random_control
from quompiler.utils.std_decompose import ctrl_decompose

formatter = MatrixFormatter(precision=2)


def test_ctr_decompose():
    k = random.randint(3, 8)
    t = random.randint(1, 3)
    ctrls = random_control(k, t)
    cu = random_CtrlGate(ctrls)
    # print(cu.controller)
    factory = FactoryManager().create_factory()
    device = factory.get_device()
    actual = ctrl_decompose(cu, device=device, clength=1)
    assert len(actual) == 5
