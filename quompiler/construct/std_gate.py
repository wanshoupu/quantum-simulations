import copy
from typing import Union, Sequence

from numpy.typing import NDArray

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qontroller import Qontroller
from quompiler.construct.qspace import  Qubit
from quompiler.construct.types import QType, UnivGate


class CtrlStdGate:
    """
    Represent a controlled standard gate operation, namely,
    a single-qubit unitary matrix in one of the UnivGate along with a control sequence.
    See also: :class:`cgate.CtrlGate`
    """

    def __init__(self, gate: UnivGate, control: Union[Sequence[QType], Qontroller], qspace: Sequence[Qubit] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param gate: the core matrix.
        :param control: the control sequence or a Qontroller.
        Dimension of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        """
        self._ctrlgate: CtrlGate = CtrlGate(gate.matrix, control, qspace)
        self.gate: UnivGate = gate

    def get_controller(self):
        return self._ctrlgate.controls

    def get_qubits(self):
        return self._ctrlgate.qids()

    def __repr__(self):
        return f'{repr(self.gate)},{repr(self._ctrlgate)}'

    def inflate(self) -> NDArray:
        return self._ctrlgate.inflate()

    def isid(self) -> bool:
        return self.gate == UnivGate.I

    def is_ctrl_singlet(self) -> bool:
        return 1 == len(self._ctrlgate.control_qids())

    def __matmul__(self, other: 'CtrlStdGate') -> CtrlGate:
        return self._ctrlgate @ other._ctrlgate

    def __copy__(self):
        return CtrlStdGate(self.gate, self._ctrlgate.controls, self._ctrlgate.qspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        gate: UnivGate = copy.deepcopy(self.gate, memodict),
        controlledM: CtrlGate = copy.deepcopy(self._ctrlgate, memodict),
        new = CtrlStdGate(gate, controlledM.controls, controlledM.qspace)
        memodict[self] = new
        return new

    def to_ctrlgate(self) -> CtrlGate:
        return self._ctrlgate

    @classmethod
    def convert(cls, cm: CtrlGate) -> 'CtrlStdGate':
        assert cm.is_std()
        result = UnivGate.get(cm._unitary.matrix)
        return CtrlStdGate(result, cm.controls, cm.qspace)

    def sorted(self, sorting: Sequence[int] = None) -> 'CtrlStdGate':
        """
        Create a sorted version of this ControlledGate.
        Sorting the ControlledGate means to sort the qspace in ascending order.
        Unless the qspace was originally sorted in this order, this necessarily incurs changes in other parts such as control sequence.
        This latter will in turn change the core indexes in the field `unitary`.
        :return: A sorted version of this ControlledGate whose qspace is in ascending order. If this is already sorted, return self.
        """
        sorted_gate: CtrlGate = self._ctrlgate.sorted(sorting)
        return CtrlStdGate(self.gate, sorted_gate.controls, sorted_gate.qspace)

    def control_qids(self) -> list[int]:
        return self._ctrlgate.control_qids()

    def target_qids(self) -> list[int]:
        return self._ctrlgate.target_qids()

    def expand(self, qspace: Sequence[Qubit]) -> CtrlGate:
        """
        Expand into the super qspace by adding necessary dimensions.
        :param qspace: the super qspace that must be a cover of self.qspace and must be sorted in ascending order.
        :return: a ControlledGate
        """
        return self.to_ctrlgate().expand(qspace)
