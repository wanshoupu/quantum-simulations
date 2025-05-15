import copy
from itertools import chain
from typing import Union, Sequence

import numpy as np
from numpy import kron
from numpy.typing import NDArray

from quompiler.construct.qontroller import Qontroller
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.shuffle import Shuffler


class CtrlGate:
    """
    Represent a controlled unitary operation with a control sequence and an n-qubit unitary matrix.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, gate: Union[UnivGate, NDArray], control: Union[Sequence[QType], Qontroller], qspace: Sequence[Qubit] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param mat: the core matrix operation. It may be a NDArray or UnivGate.
        :param control: the control sequence or a Qontroller. Order of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        """
        self._controller = control if isinstance(control, Qontroller) else Qontroller(control)
        core = self._controller.core()
        mat = gate.matrix if isinstance(gate, UnivGate) else gate
        assert mat.shape[0] == len(core), f'matrix shape does not match the control sequence'
        dim = 1 << self._controller.length
        self.unitary = UnitaryM(dim, core, mat)
        self.gate = gate if isinstance(gate, UnivGate) else UnivGate.get(gate)

        if qspace is None:
            self.qspace = [Qubit(i) for i in range(self._controller.length)]
        else:
            # qspace is a sequence
            assert len(qspace) == len(set(qspace))  # uniqueness
            assert len(qspace) == len(control)  # consistency
            self.qspace = list(qspace)

    def __repr__(self):
        return f'{repr(self.unitary)},controls={repr(self._controller.controls)},qspace={self.qspace}'

    def order(self) -> int:
        return self.unitary.order()

    def inflate(self) -> NDArray:
        return self.unitary.inflate()

    def isid(self) -> bool:
        return self.unitary.isid()

    def is_std(self) -> bool:
        return self.gate is not None

    def is2l(self) -> bool:
        return self.unitary.is2l()

    def issinglet(self) -> bool:
        """
        Check if this CtrlGate only operates on a single-qubit
        :return: True if this CtrlGate only operates on a single-qubit. False otherwise.
        """
        return len(self.target_qids()) == 1

    def __matmul__(self, other: 'CtrlGate') -> 'CtrlGate':
        # direct product
        if self.qspace == other.qspace:
            unitary = self.unitary @ other.unitary
            return CtrlGate.convert(unitary, self.qspace)
        # sort qspace
        if set(self.qspace) == set(other.qspace):
            return self.sorted() @ other.sorted()
        # expand to same qspace
        univ = set(self.qspace + other.qspace)
        return self.expand(list(univ - set(self.qspace))) @ other.expand(list(univ - set(other.qspace)))

    def __copy__(self):
        return CtrlGate(self.unitary.matrix, self._controller.controls, self.qspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        new = CtrlGate(copy.deepcopy(self.unitary.matrix, memodict), copy.deepcopy(self._controller.controls, memodict), copy.deepcopy(self.qspace, memodict))
        memodict[self] = new
        return new

    @classmethod
    def convert(cls, u: UnitaryM, qspace: Sequence[Qubit] = None) -> 'CtrlGate':
        """
        Convert a UnitaryM to CtrlGate based on organically grown control sequence.
        This can potentially expand the order of matrix to a number that is power of 2.
        :param u:
        :param qspace:
        :return:
        """
        if qspace is None:
            assert u.order() & (u.order() - 1) == 0
            n = u.order().bit_length() - 1
            qspace = [Qubit(i) for i in range(n)]
        else:
            assert len(qspace) == len(set(qspace))  # uniqueness
            n = len(qspace)
            assert u.order() == 1 << n

        controller = Qontroller.create(n, u.core)
        core = controller.core()
        # assert set(u.core) <= set(core)
        lookup = {idx: i for i, idx in enumerate(core)}
        indxs = [lookup[c] for c in u.core]
        m = np.eye(len(core), dtype=np.complexfloating)
        m[np.ix_(indxs, indxs)] = u.matrix
        return CtrlGate(m, controller, qspace=qspace)

    def sorted(self, sorting: Sequence[Qubit] = None) -> 'CtrlGate':
        """
        Create a sorted version of this CtrlGate.
        Sorting the CtrlGate means to sort the qubits according to the given sorting order and at the same time transform the operator
        such that the operations on all qubits remain invariant.
        :param sorting: optional sorting sequence. If not provided, will sort by qspace
        :return: A sorted version of this CtrlGate whose qspace is in ascending order. If this is already sorted, return self.
        """
        if sorting is None:
            sorting = np.argsort(self.qspace)
        else:
            assert list(range(self._controller.length)) == sorted(sorting)

        # create the sorted controls
        controls = [self._controller[i] for i in sorting]
        # create the sorted qspace
        qspace = [self.qspace[i] for i in sorting]

        # create the sorted core matrix
        new_targets = [qid for i, qid in enumerate(qspace) if controls[i] == QType.TARGET]
        sh = Shuffler.from_permute(self.target_qids(), new_targets)
        indexes = sh.map_all(range(1 << len(new_targets)))
        mat = self.unitary.matrix[np.ix_(indexes, indexes)]
        return CtrlGate(mat, controls, qspace)

    def qids(self) -> list[Qubit]:
        return self.qspace

    def control_qids(self) -> list[Qubit]:
        target = self.target_qids()
        return [qid for qid in self.qspace if qid not in target]

    def target_qids(self) -> list[Qubit]:
        return [qid for i, qid in enumerate(self.qspace) if self._controller.controls[i] == QType.TARGET]

    def project(self, qspace: Sequence[Qubit]) -> 'CtrlGate':
        """
        Use with CAUTION: This operation may lead to incorrect results.
        Project a CtrlGate to its subsystem represented by a subset of qspace.
        TODO: a better approach would be to use FactorMat as core matrix for CtrlGate. Then this operation would be made safe - to eliminate IDLER qubits only.
        :param qspace: must be a subset of self.qspace and must be sorted in ascending order. If the qspace is identical to self.qspace, noop.
        :return: a projected CtrlGate.
        """
        raise NotImplementedError('This method is not implemented.')

    def expand(self, qspace: Sequence[Qubit], ctrls: Sequence[QType] = None) -> 'CtrlGate':
        """
        Expand into the new qspace with additional qubits with the optional control sequence.
        :param qspace: the qspace subtended by the additional qubits.
        :param ctrls: the optional control sequence to be used for each of the additional qubits. If not provided, will use QType.TARGET for all new qubits.
        :return: a CtrlGate
        """
        if not qspace:
            return self
        assert not set(qspace) & set(self.qspace), f'Extended qspace must not overlap with existing qspace.'
        if ctrls is None:
            # TODO treat IDLER qubits by default.
            ctrls = [QType.TARGET] * len(qspace)
        else:
            assert all(c != QType.IDLER for c in ctrls)
            assert len(qspace) == len(ctrls)

        mat = self.unitary.matrix
        for i, qid in enumerate(qspace):
            if ctrls[i] == QType.TARGET:
                mat = kron(mat, np.eye(2))
        extended_ctrl = list(chain(self._controller, ctrls))
        extended_qspace = list(chain(self.qspace, qspace))
        return CtrlGate(mat, extended_ctrl, extended_qspace)
