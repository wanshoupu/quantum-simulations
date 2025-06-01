import cProfile
import random
from abc import ABC, abstractmethod

import numpy as np
import pytest

from quompiler.construct.su2net import SU2Net
from quompiler.construct.types import SU2NetType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary

formatter = MatrixFormatter(precision=2)

error_margin = .25


@pytest.mark.skip(reason="These tests are for manual run only (comment out this line to run and do not commit changes)")
class PerfTestSu2Net(ABC):
    su2net: SU2Net = None

    @abstractmethod
    def class_name(self):
        return self.__name__

    def test_lookup_error_margin(self):
        print(f"\nTest class: {self.class_name()}")
        pr = cProfile.Profile()
        pr.enable()
        for seed in random.sample(range(1 << 20), 1000):
            random.seed(seed)
            np.random.seed(seed)

            u = random_unitary(2)

            # execute
            _, lookup_error = self.su2net.lookup(u)
            # print(f'\n{formatter.tostr(u)}\nlookup error: {lookup_error}, dist error: {lookup_error}')

        pr.disable()
        pr.print_stats(sort='time')


class TestAutoNN(PerfTestSu2Net):
    su2net: SU2Net = SU2Net(error_margin, SU2NetType.AutoNN)

    def class_name(self):
        return 'AutoNN'


class TestBruteNN(PerfTestSu2Net):
    su2net: SU2Net = SU2Net(error_margin, SU2NetType.BruteNN)

    def class_name(self):
        return 'BruteNN'


class TestBallTreeNN(PerfTestSu2Net):
    su2net: SU2Net = SU2Net(error_margin, SU2NetType.BallTreeNN)

    def class_name(self):
        return 'BallTreeNN'


class TestKDTreeNN(PerfTestSu2Net):
    su2net: SU2Net = SU2Net(error_margin, SU2NetType.KDTreeNN)

    def class_name(self):
        return 'KDTreeNN'
