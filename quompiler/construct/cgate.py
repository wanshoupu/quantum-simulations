import copy
from typing import Union, Sequence

import numpy as np
from numpy import kron
from numpy.typing import NDArray

from quompiler.construct.qontroller import core2control, ctrl2core
from quompiler.construct.qspace import Qubit
from quompiler.construct.su2gate import RGate
from quompiler.construct.types import QType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.inter_product import ctrl_expand, qproject, is_idler
from quompiler.utils.mfun import allprop, herm
from quompiler.utils.permute import Permuter


class CtrlGate:
    """
    Represent a controlled unitary operation with a control sequence and an n-qubit unitary matrix.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, gate: Union[UnivGate, NDArray, RGate], control: Sequence[QType], qspace: Sequence[Qubit] = None, phase: complex = 1.0):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param gate: the core matrix operation. It may be a NDArray or UnivGate or RGate.
        :param control: the control sequence or a Qontroller. Order of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        """
        assert all(isinstance(c, QType) for c in control), f'control contains non-QType items'
        core = ctrl2core(control)
        mat = np.array(gate)
        self.gate = UnivGate.get_prop(mat)
        if self.gate is not None:
            phase *= allprop(mat, np.array(self.gate)).result
            mat = np.array(self.gate)
        elif isinstance(gate, RGate):
            self.gate = gate
        assert mat.shape[0] == len(core), f'matrix shape does not match the control sequence'
        dim = 1 << len(control)
        self._unitary = UnitaryM(dim, core, mat, phase=phase)

        if qspace is None:
            qspace = [Qubit(i) for i in range(len(control))]
        else:
            # qspace is a sequence
            assert all(isinstance(q, Qubit) for q in qspace), f'qspace contains non-Qubit items'
            assert len(qspace) == len(set(qspace))  # uniqueness
            assert len(qspace) == len(control)  # consistency
        self.qspace = list(qspace)
        self._qontrol = {q: c for q, c in zip(qspace, control)}

    def __repr__(self):
        gate = self.gate if self.gate else self._unitary.matrix
        return f'CtrlGate{{{repr(gate)},controls={repr(self.controls())},qspace={repr(self.qspace)}}}'

    def order(self) -> int:
        return self._unitary.order()

    def inflate(self) -> NDArray:
        return self._unitary.inflate()

    def __array__(self) -> NDArray:
        return self.inflate()

    def isid(self) -> bool:
        return self._unitary.isid()

    def is_std(self) -> bool:
        return isinstance(self.gate, UnivGate)

    def is_principal(self) -> bool:
        return isinstance(self.gate, RGate) and self.gate.axis.principal

    def is2l(self) -> bool:
        return self._unitary.is2l()

    def issinglet(self) -> bool:
        """
        Check if this CtrlGate only operates on a single-qubit
        :return: True if this CtrlGate only operates on a single-qubit. False otherwise.
        """
        return len(self.target_qids()) == 1

    def core(self) -> tuple[int]:
        return self._unitary.core

    def matrix(self) -> NDArray:
        return self._unitary.matrix

    def phase(self):
        return self._unitary.phase

    def controls(self) -> list[QType]:
        return [self._qontrol[q] for q in self.qspace]

    def qontrol(self) -> dict[Qubit, QType]:
        return self._qontrol

    def herm(self) -> 'CtrlGate':
        return CtrlGate(herm(self.matrix()), self.controls(), self.qspace, self.phase())

    def __matmul__(self, other: 'CtrlGate') -> 'CtrlGate':
        if not isinstance(other, CtrlGate):
            return NotImplemented(f'cannot matmul with {type(other)}')
        # direct product
        if self.qspace == other.qspace and self.controls() == other.controls():
            return CtrlGate(self.matrix() @ other.matrix(), self.controls(), self.qspace, self.phase() * other.phase())
        # sort qspace
        if self._qontrol == other._qontrol:
            return self.sorted() @ other.sorted()
        # resolve ctrl conflict
        conflicts = {q for q in self._qontrol.keys() & other._qontrol.keys() if self._qontrol[q] != other._qontrol[q]}
        if conflicts:
            return self.promote(list(conflicts)) @ other.promote(list(conflicts))
        # expand to same qspace
        univ = self._qontrol.keys() | other._qontrol.keys()
        qspace1 = list(univ - set(self.qspace))
        qspace2 = list(univ - set(other.qspace))
        return self.expand(qspace1) @ other.expand(qspace2)

    def __copy__(self):
        return CtrlGate(self.matrix(), self.controls(), self.qspace, self.phase())

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        new = CtrlGate(copy.deepcopy(self.matrix(), memodict), copy.deepcopy(self.controls(), memodict), copy.deepcopy(self.qspace, memodict), copy.deepcopy(self.phase()))
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
            # assert sorting is valid over qspace
            assert list(range(len(self.qspace))) == sorted(sorting)

        # create the sorted qspace
        qspace = [self.qspace[i] for i in sorting]
        # create the sorted controls
        controls = [self._qontrol[q] for q in qspace]

        # create the sorted core matrix
        new_targets = [qid for i, qid in enumerate(qspace) if controls[i] == QType.TARGET]
        perm = Permuter.from_permute(self.target_qids(), new_targets)
        indexes = perm.bitsortall(range(1 << len(new_targets)))
        mat = self.matrix()[np.ix_(indexes, indexes)]
        return CtrlGate(mat, controls, qspace, self.phase())

    def qids(self) -> list[Qubit]:
        return self.qspace

    def control_qids(self) -> list[Qubit]:
        return [qid for qid in self.qspace if self._qontrol[qid] in QType.CONTROL1 | QType.CONTROL0]

    def target_qids(self) -> list[Qubit]:
        return [qid for qid in self.qspace if self._qontrol[qid] == QType.TARGET]

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

        mat = self.matrix()
        for i, qid in enumerate(qspace):
            if ctrls[i] == QType.TARGET:
                mat = kron(mat, np.eye(2))
        extended_ctrl = self.controls() + list(ctrls)
        extended_qspace = self.qspace + list(qspace)
        return CtrlGate(mat, extended_ctrl, extended_qspace, self.phase())

    def is_idler(self, qubit) -> bool:
        if qubit not in self._qontrol:
            return True
        if self._qontrol[qubit] == QType.CONTROL1:
            return False
        idx = self.target_qids().index(qubit)
        return is_idler(self.matrix(), idx)

    def trace(self, qubit: Qubit, state: NDArray) -> 'CtrlGate':
        """
        Trace out a subsystem consisting the target qubit with a given pure state |ψ⟩ and compute the CtrlGate for the remaining qubits.

        This computes the effective operator on the remaining qubits by sandwiching U with |ψ⟩ on the specified qubit.
        That is, it returns:
            U_eff = ⟨ψ| U |ψ⟩

        Parameters:
        -----------
        :param state: np.ndarray, state to be used to trace out the qubit.
            A 2-dimensional complex vector representing a normalized pure state |ψ⟩ of a single qubit.

        :param qubit:
            The qubit subsystem to be traced out, where |ψ⟩ lives.
            Note that our matrix is based on [qubit-0 ⨂ qubit-1 ⨂ ...].

        Returns:
        --------
        U_eff : CtrlGate representing the traced operator acting on the remaining n-1 qubits.

        Notes:
        ------
        - This is not the same as a partial trace. It conditions on the subsystem being in a known state |ψ⟩.
        - Useful for computing effective dynamics or gates when a subsystem is initialized or post-selected in a known pure state.

        Example:
        --------
         ψ = np.array([1, 0])  # |0>
         U = CtrlGate(...)  # |01>
         U_eff = U.trace(ψ, qubit)
        """
        assert qubit in self._qontrol, f'Qubit {qubit} not in qspace.'
        assert state.shape == (2,), f'state vector must be a 1D array of length 2, but got {state.shape}.'
        assert np.isclose(np.linalg.norm(state), 1), f'state vector must normalized but got {state}.'

        ctrl = self._qontrol[qubit]
        if ctrl == QType.IDLER:
            # nothing need to be changed but to eliminate the qubit
            raise NotImplementedError(f"CtrlGate {ctrl} not implemented.")
            # return CtrlGate(self.matrix(), new_ctrls, new_qspace)

        idx = self.qspace.index(qubit)
        new_qspace = self.qspace[:idx] + self.qspace[idx + 1:]
        new_ctrls = [self._qontrol[q] for q in new_qspace]
        if ctrl in QType.CONTROL1 | QType.CONTROL0:
            a, b = state
            u = self.matrix()
            e = np.eye(u.shape[0])
            u, e = (u, e) if ctrl == QType.CONTROL1 else (e, u)
            mat = abs(a) ** 2 * e + abs(b) ** 2 * u
            return CtrlGate(mat, new_ctrls, new_qspace, self.phase())

        assert ctrl == QType.TARGET
        tq = self.target_qids()
        if len(tq) == 1:
            return CtrlGate(UnivGate.I, [QType.TARGET], new_qspace, self.phase())

        mat = qproject(self.matrix(), tq.index(qubit), state)
        return CtrlGate(mat, new_ctrls, new_qspace, self.phase())

    def dela(self, state=None) -> 'CtrlGate':
        """
        Eliminate all ancilla qubits by projecting them to the base `state`, or [1, 0] if not given.
        Use with caution! Exception may be raised if the resulting gate is not unitary.

        :return: A CtrlGate free from ancilla qubits.
        """
        if state is None:
            state = np.array([1, 0])
        else:
            assert isinstance(state, np.ndarray), f'state vector must be a np.ndarray.'
            assert state.shape == (2,), f'state vector must be a 1D array of length 2, but got {state.shape}'
        ancillas = [q for q in self.qspace if q.ancilla]
        gate = self
        for q in ancillas:
            gate = gate.trace(q, state)
        return gate

    def promote(self, qubits: Sequence[Qubit]) -> 'CtrlGate':
        """
        Promote the control sequence of a subset of qspace to QType.TARGET, if not already, from other QTypes.
        :param qubits: subset of the qspace to promote.
        :return: a CtrlGate with promoted control sequence.
        """
        if not qubits or all(self._qontrol[q] == QType.TARGET for q in qubits):
            return self
        qubits = set(qubits)
        assert qubits <= set(self.qspace), f'Promoting qubits must be a subset of existing qspace.'
        mat = self.matrix() * self.phase()
        target_count = 0
        new_ctrls = []
        for q in reversed(self.qspace):
            ctrl = self._qontrol[q]
            if q in qubits and ctrl in QType.CONTROL0 | QType.CONTROL1:
                mat = ctrl_expand(mat, 1 << target_count, ctrl.base[0])
                target_count += 1
                new_ctrls.append(QType.TARGET)
            elif ctrl == QType.TARGET:
                target_count += 1
                new_ctrls.append(QType.TARGET)
            else:
                new_ctrls.append(ctrl)
        return CtrlGate(mat, new_ctrls[::-1], self.qspace)
