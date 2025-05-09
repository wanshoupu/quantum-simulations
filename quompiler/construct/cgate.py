import copy
from itertools import groupby, accumulate
from typing import Union, Sequence

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.qontroller import Qontroller
from quompiler.construct.qspace import QSpace
from quompiler.construct.types import QType, UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.inter_product import mesh_product


class CtrlGate:
    """
    Represent a controlled unitary operation with a control sequence and an n-qubit unitary matrix.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, m: NDArray, control: Union[Sequence[QType], Qontroller], qspace: Union[Sequence[int], QSpace] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param m: the core matrix.
        :param control: the control sequence or a Qontroller.
        Dimension of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        """
        self.controller = control if isinstance(control, Qontroller) else Qontroller(control)
        core = self.controller.core()
        assert m.shape[0] == len(core)
        dim = 1 << self.controller.length
        self.unitary = UnitaryM(dim, core, m)
        if qspace is None:
            self.qspace = QSpace(list(range(self.controller.length)))
        elif isinstance(qspace, QSpace):
            self.qspace = qspace
        else:
            # qspace is a sequence
            assert len(qspace) == len(set(qspace))  # uniqueness
            assert len(qspace) == len(control)  # consistency
            self.qspace = QSpace(qspace)

    def __repr__(self):
        return f'{repr(self.unitary)},controls={repr(self.controller.controls)},qspace={self.qspace}'

    def inflate(self) -> NDArray:
        res = self.unitary.inflate()
        if self.is_sorted():
            return res
        indexes = self.qspace.map_all(range(res.shape[0]))
        return res[np.ix_(indexes, indexes)]

    def isid(self) -> bool:
        return self.unitary.isid()

    def is_std(self) -> bool:
        return UnivGate.get(self.unitary.matrix) is not None

    def is2l(self) -> bool:
        return self.unitary.is2l()

    def issinglet(self) -> bool:
        """
        Check if this ControlledGate only operates on a single-qubit
        :return: True if this ControlledGate only operates on a single-qubit. False otherwise.
        """
        return len(self.target_qids()) == 1

    def __matmul__(self, other: 'CtrlGate') -> 'CtrlGate':
        if self.qspace == other.qspace:
            unitary = self.unitary @ other.unitary
            return CtrlGate.convert(unitary, self.qspace)
        univ = sorted(set(self.qspace.qids + other.qspace.qids))
        return self.expand(univ) @ other.expand(univ)

    def __copy__(self):
        return CtrlGate(self.unitary.matrix, self.controller.controls, self.qspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        new = CtrlGate(copy.deepcopy(self.unitary.matrix, memodict), copy.deepcopy(self.controller.controls, memodict), copy.deepcopy(self.qspace, memodict))
        memodict[self] = new
        return new

    @classmethod
    def convert(cls, u: UnitaryM, qspace: Union[Sequence[int], QSpace] = None) -> 'CtrlGate':
        """
        Convert a UnitaryM to ControlledGate based on organically grown control sequence.
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

    def is_sorted(self) -> bool:
        return self.qspace.is_sorted()

    def sorted(self):
        """
        Create a sorted version of this ControlledGate.
        Sorting the ControlledGate means to sort the qspace in ascending order.
        Unless the qspace was originally sorted in this order, this necessarily incurs changes in other parts such as control sequence.
        This latter will in turn change the core indexes in the field `unitary`.
        :return: A sorted version of this ControlledGate whose qspace is in ascending order. If this is already sorted, return self.
        """
        if self.is_sorted():
            return self

        # prepare the controls
        sorting = np.argsort(self.qspace.qids)
        controls = [self.controller.controls[i] for i in sorting]
        # create the sorted ControlledGate
        return CtrlGate(self.unitary.matrix, controls, sorted(self.qspace.qids))

    def control_qids(self) -> list[int]:
        target = self.target_qids()
        return [qid for qid in self.qspace.qids if qid not in target]

    def target_qids(self) -> list[int]:
        return [qid for i, qid in enumerate(self.qspace.qids) if self.controller.controls[i] == QType.TARGET]

    def expand(self, qspace: Union[QSpace, Sequence[int]]) -> 'CtrlGate':
        """
        Expand into the super qspace by adding necessary dimensions.
        :param qspace: the super qspace that must be a cover of self.qspace and must be sorted in ascending order.
        :return: a ControlledGate
        """
        if not isinstance(qspace, QSpace):
            qspace = QSpace(qspace)
        assert all(qspace.qids[i - 1] < qspace.qids[i] for i in range(1, qspace.length))
        assert set(self.qspace.qids) <= set(qspace.qids)
        scu = self.sorted()
        if scu.qspace.qids == qspace.qids:
            return scu
        extended_tids = sorted(set(qspace.qids) - set(scu.control_qids()))
        mat = scu._calc_mat(scu, extended_tids)

        # prepare the new control sequence
        controls = [QType.TARGET] * qspace.length
        for i, qid in enumerate(scu.qspace.qids):
            j = qspace.qids.index(qid)
            controls[j] = scu.controller.controls[i]
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
