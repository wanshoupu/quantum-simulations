import copy
from itertools import groupby, accumulate
from typing import Union, Sequence

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.qontroller import Qontroller
from quompiler.construct.qspace import QSpace, Qubit
from quompiler.construct.types import QType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.inter_product import mesh_product


class CtrlGate:
    """
    Represent a controlled unitary operation with a control sequence and an n-qubit unitary matrix.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, gate: Union[UnivGate, NDArray], control: Union[Sequence[QType], Qontroller], qspace: Union[Sequence[Qubit], QSpace] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param mat: the core matrix operation. It may be a NDArray or UnivGate.
        :param control: the control sequence or a Qontroller. Order of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        """
        self._controller = control if isinstance(control, Qontroller) else Qontroller(control)
        core = self._controller.core()
        mat = gate.matrix if isinstance(gate, UnivGate) else gate
        assert mat.shape[0] == len(core)
        dim = 1 << self._controller.length
        self.unitary = UnitaryM(dim, core, mat)
        self.gate = gate if isinstance(gate, UnivGate) else UnivGate.get(gate)

        if qspace is None:
            qubits = [Qubit(i) for i in range(self._controller.length)]
            self.qspace = QSpace(qubits)
        elif isinstance(qspace, QSpace):
            self.qspace = qspace
        else:
            # qspace is a sequence
            assert len(qspace) == len(set(qspace))  # uniqueness
            assert len(qspace) == len(control)  # consistency
            self.qspace = QSpace(qspace)

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
        if self.qspace == other.qspace:
            unitary = self.unitary @ other.unitary
            return CtrlGate.convert(unitary, self.qspace)
        univ = sorted(set(self.qspace.qids + other.qspace.qids))
        return self.expand(univ) @ other.expand(univ)

    def __copy__(self):
        return CtrlGate(self.unitary.matrix, self._controller.controls, self.qspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        new = CtrlGate(copy.deepcopy(self.unitary.matrix, memodict), copy.deepcopy(self._controller.controls, memodict), copy.deepcopy(self.qspace, memodict))
        memodict[self] = new
        return new

    @classmethod
    def convert(cls, u: UnitaryM, qspace: Union[Sequence[Qubit], QSpace] = None) -> 'CtrlGate':
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
            qspace = QSpace(range(n))
        elif not isinstance(qspace, QSpace):
            # qspace is a sequence
            assert len(qspace) == len(set(qspace))  # uniqueness
            qspace = QSpace(qspace)
            n = qspace.length
            assert u.order() == 1 << n
            assert u.core in set(qspace.qids)
        else:
            n = qspace.length

        controller = Qontroller.create(n, u.core)
        assert qspace.length == len(controller.controls)  # consistency
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
            sorting = np.argsort(self.qspace.qids)
        else:
            assert self._controller.length == len(sorting) == len(set(sorting))
        controls = [self._controller[i] for i in sorting]
        # create the sorted CtrlGate
        qspace = [self.qspace[i] for i in sorting]
        return CtrlGate(self.unitary.matrix, controls, qspace)

    def qids(self) -> list[Qubit]:
        return self.qspace.qids

    def control_qids(self) -> list[Qubit]:
        target = self.target_qids()
        return [qid for qid in self.qspace.qids if qid not in target]

    def target_qids(self) -> list[Qubit]:
        return [qid for i, qid in enumerate(self.qspace.qids) if self._controller.controls[i] == QType.TARGET]

    def project(self, qspace: Union[QSpace, Sequence[Qubit]]) -> 'CtrlGate':
        """
        Use with CAUTION: This operation may lead to incorrect results.
        Project a CtrlGate to its subsystem represented by a subset of QSpace.
        TODO: a better approach would be to use FactorMat as core matrix for CtrlGate. Then this operation would be made safe - to eliminate IDLER qubits only.
        :param qspace: must be a subset of self.qspace and must be sorted in ascending order. If the qspace is identical to self.qspace, noop.
        :return: a projected CtrlGate.
        """
        pass

    def extend(self, qspace: Sequence[Qubit], ctrls: Sequence[QType] = None) -> 'CtrlGate':
        """
        Extend the control sequence to new qubits.
        :param qspace: The extra qspace to be extended to.
        :param ctrls: The control sequence to be applied. If not provided, defaults to [QType.CONTROL1] * len(qspace).
        :return: A new CtrlGate with extended control sequence.
        """
        assert not set(qspace) & set(self.qspace)
        ctrls = list(ctrls) if ctrls else [QType.CONTROL1] * len(qspace)
        assert all(c in QType.CONTROL0 | QType.CONTROL1 for c in ctrls)  # only CONTROLs are present

        newctrl = Qontroller(list(self._controller) + ctrls)
        newqspace = QSpace(list(self.qspace) + list(qspace))
        return CtrlGate(self.unitary.matrix, newctrl, newqspace)

    def expand(self, qspace: Union[QSpace, Sequence[Qubit]]) -> 'CtrlGate':
        """
        Expand into the superset qspace by adding IDLER qubits. TODO treat IDLER qubits the same as TARGET.
        :param qspace: the super qspace that must be a cover of self.qspace and must be sorted in ascending order.
        :return: a CtrlGate
        """
        if not isinstance(qspace, QSpace):
            qspace = QSpace(qspace)
        assert qspace.sorting == list(range(qspace.length))
        assert set(self.qspace.qids) <= set(qspace.qids)
        scu = self.sorted()
        if scu.qspace == qspace:
            return scu
        extended_tids = sorted(set(qspace) - set(scu.control_qids()))
        mat = scu._calc_mat(scu, extended_tids)

        # prepare the new control sequence
        controls = [QType.TARGET] * qspace.length
        for i, qid in enumerate(scu.qspace.qids):
            j = qspace.qids.index(qid)
            controls[j] = scu._controller.controls[i]
        return CtrlGate(mat, controls, qspace)

    @staticmethod
    def _calc_mat(cu, extended_ids):
        core_ids = set(cu.target_qids())
        labels = [x in core_ids for x in extended_ids]
        counts = [(k, sum(1 for _ in g)) for k, g in groupby(labels)]
        matrices = [cu.unitary.matrix] + [np.eye(1 << c) for k, c in counts if not k]

        skips = list(accumulate([c for k, c in counts if k]))
        if not counts[0][0]:
            skips = [0] + skips
        if len(skips) < len(matrices):
            skips.append(len(core_ids))
        partitions = [1 << n for n in skips]
        assert len(matrices) == len(partitions)
        return mesh_product(matrices, partitions)
