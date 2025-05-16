import copy
from itertools import chain
from typing import Union, Sequence

import numpy as np
from numpy import kron
from numpy.typing import NDArray

from quompiler.construct.qontroller import core2control, ctrl2core
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.inter_product import block_ctrl
from quompiler.utils.shuffle import Shuffler


class CtrlGate:
    """
    Represent a controlled unitary operation with a control sequence and an n-qubit unitary matrix.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, gate: Union[UnivGate, NDArray], control: Sequence[QType], qspace: Sequence[Qubit] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param mat: the core matrix operation. It may be a NDArray or UnivGate.
        :param control: the control sequence or a Qontroller. Order of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        """
        self.controls = control
        core = ctrl2core(control)
        mat = gate.matrix if isinstance(gate, UnivGate) else gate
        assert mat.shape[0] == len(core), f'matrix shape does not match the control sequence'
        dim = 1 << len(self.controls)
        self._unitary = UnitaryM(dim, core, mat)
        self.gate = gate if isinstance(gate, UnivGate) else UnivGate.get(gate)

        if qspace is None:
            qspace = [Qubit(i) for i in range(len(self.controls))]
        else:
            # qspace is a sequence
            assert len(qspace) == len(set(qspace))  # uniqueness
            assert len(qspace) == len(control)  # consistency
        self.qspace = list(qspace)
        self._qlookup = {q: c for q, c in zip(qspace, control)}

    def __repr__(self):
        return f'{repr(self._unitary)},controls={repr(self.controls)},qspace={self.qspace}'

    def order(self) -> int:
        return self._unitary.order()

    def inflate(self) -> NDArray:
        return self._unitary.inflate()

    def isid(self) -> bool:
        return self._unitary.isid()

    def is_std(self) -> bool:
        return self.gate is not None

    def is2l(self) -> bool:
        return self._unitary.is2l()

    def issinglet(self) -> bool:
        """
        Check if this CtrlGate only operates on a single-qubit
        :return: True if this CtrlGate only operates on a single-qubit. False otherwise.
        """
        return len(self.target_qids()) == 1

    def __matmul__(self, other: 'CtrlGate') -> 'CtrlGate':
        if not isinstance(other, CtrlGate):
            return NotImplemented(f'cannot matmult with {type(other)}')
        # direct product
        if self.qspace == other.qspace and self.controls == other.controls:
            return CtrlGate(self._unitary.matrix @ other._unitary.matrix, other.controls, self.qspace)
        # sort qspace
        if self._qlookup == other._qlookup:
            return self.sorted() @ other.sorted()
        # resolve ctrl conflict
        conflicts = {q for q in self._qlookup.keys() & other._qlookup.keys() if self._qlookup[q] != other._qlookup[q]}
        if conflicts:
            return self.promote(list(conflicts)) @ other.promote(list(conflicts))
        # expand to same qspace
        univ_lookup = {q: self._qlookup.get(q) or other._qlookup[q] for q in self._qlookup.keys() | other._qlookup.keys()}
        qspace1 = list(univ_lookup.keys() - set(self.qspace))
        ctrl1 = [univ_lookup[q] for q in qspace1]
        qspace2 = list(univ_lookup.keys() - set(other.qspace))
        ctrl2 = [univ_lookup[q] for q in qspace2]
        return self.expand(qspace1, ctrl1) @ other.expand(qspace2, ctrl2)

    def __copy__(self):
        return CtrlGate(self._unitary.matrix, self.controls, self.qspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        new = CtrlGate(copy.deepcopy(self._unitary.matrix, memodict), copy.deepcopy(self.controls, memodict), copy.deepcopy(self.qspace, memodict))
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

        controls = core2control(n, u.core)
        core = ctrl2core(controls)
        # assert set(u.core) <= set(core)
        lookup = {idx: i for i, idx in enumerate(core)}
        indxs = [lookup[c] for c in u.core]
        m = np.eye(len(core), dtype=np.complexfloating)
        m[np.ix_(indxs, indxs)] = u.matrix
        return CtrlGate(m, controls, qspace=qspace)

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
            assert list(range(len(self.controls))) == sorted(sorting)

        # create the sorted controls
        controls = [self.controls[i] for i in sorting]
        # create the sorted qspace
        qspace = [self.qspace[i] for i in sorting]

        # create the sorted core matrix
        new_targets = [qid for i, qid in enumerate(qspace) if controls[i] == QType.TARGET]
        sh = Shuffler.from_permute(self.target_qids(), new_targets)
        indexes = sh.map_all(range(1 << len(new_targets)))
        mat = self._unitary.matrix[np.ix_(indexes, indexes)]
        return CtrlGate(mat, controls, qspace)

    def qids(self) -> list[Qubit]:
        return self.qspace

    def control_qids(self) -> list[Qubit]:
        target = self.target_qids()
        return [qid for qid in self.qspace if qid not in target]

    def target_qids(self) -> list[Qubit]:
        return [qid for i, qid in enumerate(self.qspace) if self.controls[i] == QType.TARGET]

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

        mat = self._unitary.matrix
        for i, qid in enumerate(qspace):
            if ctrls[i] == QType.TARGET:
                mat = kron(mat, np.eye(2))
        extended_ctrl = list(chain(self.controls, ctrls))
        extended_qspace = list(chain(self.qspace, qspace))
        return CtrlGate(mat, extended_ctrl, extended_qspace)

    def promote(self, qubits: Sequence[Qubit]) -> 'CtrlGate':
        """
        Promote the control sequence of a subset of qspace to QType.TARGET, if not already, from other QTypes.
        :param qubits: subset of the qspace to promote.
        :return: a CtrlGate with promoted control sequence.
        """
        qubits = set(qubits)
        assert qubits <= set(self.qspace), f'Promoting qubits must be a subset of existing qspace.'
        mat = self._unitary.matrix
        target_count = 0
        new_ctrls = []
        for i in range(len(self.qspace) - 1, -1, -1):
            ctrl = self.controls[i]
            if self.qspace[i] in qubits and ctrl in QType.CONTROL0 | QType.CONTROL1:
                mat = block_ctrl(mat, 1 << target_count, ctrl.base[0])
                target_count += 1
                new_ctrls.append(QType.TARGET)
            elif ctrl == QType.TARGET:
                target_count += 1
                new_ctrls.append(QType.TARGET)
            else:
                new_ctrls.append(ctrl)
        return CtrlGate(mat, new_ctrls[::-1], self.qspace)
