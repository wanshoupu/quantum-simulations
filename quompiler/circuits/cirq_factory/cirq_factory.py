from typing import Dict

import cirq
from typing_extensions import override

from quompiler.circuits.abstract_factory import QFactory
from quompiler.circuits.qbuilder import CircuitBuilder
from quompiler.circuits.qdevice import QDevice
from quompiler.circuits.cirq_factory.cirq_builder import CirqBuilder
from quompiler.circuits.qompiler import Qompiler
from quompiler.config.construct import QompilerConfig, DeviceConfig
from quompiler.construct.qspace import Qubit


class CirqDevice(QDevice):

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self.qubit_lookup: Dict[Qubit, object] = {}

    @override
    def map(self, qid: Qubit):
        if qid not in self.qubit_lookup:
            self.qubit_lookup[qid] = cirq.NamedQubit(str(qid))
        return self.qubit_lookup[qid]


class CirqFactory(QFactory):

    def __init__(self, config: QompilerConfig):
        self._config = config
        self._device = None
        self._builder = None
        self._qompiler = None

    @override
    def get_qompiler(self) -> Qompiler:
        if self._qompiler is None:
            self._qompiler = Qompiler(self._config, self.get_builder(), self.get_device())
        return self._qompiler

    @override
    def get_device(self) -> QDevice:
        if self._device is None:
            self._device = CirqDevice(self._config.device)
        return self._device

    @override
    def get_builder(self) -> CircuitBuilder:
        if self._builder is None:
            self._builder = CirqBuilder(self.get_device())
        return self._builder
