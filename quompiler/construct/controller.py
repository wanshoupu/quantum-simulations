from typing import Sequence

from quompiler.construct.cmat import QubitClass


class Controller:
    def __init__(self, controls: Sequence[QubitClass]):
        self.controls = controls
        self.length = len(controls)
        self.masks = dict()
        for i, c in enumerate(controls):
            if c == QubitClass.CONTROL1:
                self.masks[self.length - 1 - i] = 1
            elif c == QubitClass.CONTROL0:
                self.masks[self.length - 1 - i] = 0

    def mask(self, n):
        for i, b in self.masks.items():
            n &= ~(1 << i)
            if b:
                n |= 1 << i
        return n

    def indexes(self):
        """
        Create the core indexes in the controlled matrix.
        Core indexes in the controlled matrix is defined to be the sparce indexes occupied by the matrix for targets + idlers.
        :return: the core indexes in the controlled matrix.
        """
        c = self.controls.count(QubitClass.CONTROL1) + self.controls.count(QubitClass.CONTROL0)
        # TODO
