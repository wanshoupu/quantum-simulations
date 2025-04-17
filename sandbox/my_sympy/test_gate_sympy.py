import numpy as np
from sympy.physics.quantum import represent
from sympy.physics.quantum.gate import XGate

if __name__ == '__main__':
    x = XGate(1)
    m = represent(x, nqubits=3)
    print(m)
    print(np.array(m))
