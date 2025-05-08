"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
This module differs from scipy.sparse in that we provide convenience specifically for quantum computer controlled unitary matrices.
"""
import copy
from typing import Union, Sequence

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.cmat import ControlledM
from quompiler.construct.qontroller import Qontroller, QSpace
from quompiler.construct.types import QType, UnivGate


class ControlledGate:
    """
    Represent a controlled standard gate operation with a control sequence and an n-qubit unitary matrix given by one of the UnivGate enum.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, gate: UnivGate, control: Union[Sequence[QType], Qontroller], qspace: Union[Sequence[int], QSpace] = None, aspace: Sequence[int] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param gate: the core matrix.
        :param control: the control sequence or a Qontroller.
        Dimension of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        :param aspace: the ancilla qubits to be used for side computation; provided in a list of integer ids. If not provided, will assume the id in the range(n) in the ancilla space.
        """
        self.controlledM: ControlledM = ControlledM(gate.matrix, control, qspace, aspace)
        self.gate: UnivGate = gate

    def __repr__(self):
        return f'{repr(self.gate)},{repr(self.controlledM)}'

    def inflate(self) -> NDArray:
        return self.controlledM.inflate()

    def isid(self) -> bool:
        return self.gate == UnivGate.I

    def __matmul__(self, other: 'ControlledGate') -> ControlledM:
        return self.controlledM @ other.controlledM

    def __copy__(self):
        return ControlledGate(self.gate, self.controlledM.controller, self.controlledM.qspace, self.controlledM.aspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        gate: UnivGate = copy.deepcopy(self.gate, memodict),
        controlledM: ControlledM = copy.deepcopy(self.controlledM, memodict),
        new = ControlledGate(gate, controlledM.controller, controlledM.qspace, controlledM.aspace)
        memodict[self] = new
        return new

    def to_controlledM(self) -> ControlledM:
        return self.controlledM

    @classmethod
    def convert(cls, cm: ControlledM) -> 'ControlledGate':
        assert cm.is_std()
        result = UnivGate.get(cm.unitary.matrix)
        return ControlledGate(result, cm.controller, cm.qspace, cm.aspace)

    def is_sorted(self) -> bool:
        return self.controlledM.is_sorted()

    def sorted(self) -> 'ControlledGate':
        """
        Create a sorted version of this ControlledGate.
        Sorting the ControlledGate means to sort the qspace in ascending order.
        Unless the qspace was originally sorted in this order, this necessarily incurs changes in other parts such as control sequence.
        This latter will in turn change the core indexes in the field `unitary`.
        :return: A sorted version of this ControlledGate whose qspace is in ascending order. If this is already sorted, return self.
        """
        if self.controlledM.is_sorted():
            return self
        controlledM: ControlledM = self.controlledM.sorted()
        return ControlledGate(self.gate, controlledM.controller, controlledM.qspace, controlledM.aspace)

    def control_qids(self) -> list[int]:
        return self.controlledM.control_qids()

    def target_qids(self) -> list[int]:
        return self.controlledM.target_qids()

    def expand(self, qspace: Union[QSpace, Sequence[int]]) -> ControlledM:
        """
        Expand into the super qspace by adding necessary dimensions.
        :param qspace: the super qspace that must be a cover of self.qspace and must be sorted in ascending order.
        :return: a ControlledGate
        """
        return self.to_controlledM().expand(qspace)
