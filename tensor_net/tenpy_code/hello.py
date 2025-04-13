import matplotlib.pyplot as plt
import numpy as np

import tenpy
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain

if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True, linewidth=100)
    plt.rcParams['figure.dpi'] = 150
    tenpy.tools.misc.setup_logging(to_stdout="INFO")
    L = 100
    model_params = {
        'J': 1., 'g': 1.,  # critical
        'L': L,
        'bc_MPS': 'finite',
    }

    M = TFIChain(model_params)
    psi = MPS.from_lat_product_state(M.lat, [['up']])
    dmrg_params = {
        'mixer': None,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10,
        },
        'verbose': True,
        'combine': True
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # the main work; modifies psi in place
    # the ground state energy was directly returned by dmrg.run()
    print("ground state energy = ", E)

    # there are other ways to extract the energy from psi:
    E1 = M.H_MPO.expectation_value(psi)  # based on the MPO
    E2 = np.sum(M.bond_energies(psi))  # based on bond terms of H, works only for a NearestNeighborModel
    assert abs(E - E1) < 1.e-10 and abs(E - E2) < 1.e-10
    # onsite expectation values

    X = psi.expectation_value("Sigmax")
    Z = psi.expectation_value("Sigmaz")
    x = np.arange(psi.L)
    plt.figure()
    plt.plot(x, Z, label="Z")
    plt.plot(x, X, label="X")  # note: it's clear that this is zero due to charge conservation!
    plt.xlabel("site")
    plt.ylabel("onsite expectation value")
    plt.legend()
    plt.show()
