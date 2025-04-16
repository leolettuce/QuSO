import argparse
from pennylane.transforms import decompose
from functools import partial
import pickle
import pennylane as qml
import numpy as np
import pyqsp
from pyqsp.angle_sequence import Polynomial, QuantumSignalProcessingPhases
import time
import os, sys

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, parent_dir)
from QuSO_utils import *

def extract_decomposed_circuit(m=1):
    epsilon=0.1
    m = 6
    full_config = [1, 1, 1, 1, 1, 1]
    # Environmental Parameters
    T_env = 293       # Ambient temperature (K)
    R_env = 0.01      # Convection resistance to ambient (K/W)

    # Heat Flows (in Watts)
    # Positive values indicate heat generation; negative values indicate cooling.
    Q_1 = 2000    
    Q_2 = 4000    
    Q_3 = -200    
    Q_4 = -2000   

    # Inter-node Thermal Resistances (in K/W)
    # These values lump together conduction and convection effects.
    R_12 = 0.005
    R_13 = 0.006
    R_14 = 0.006
    R_23 = 0.007
    R_24 = 0.007
    R_34 = 0.008

    R_dict = {
        (0, 1): R_12,
        (0, 2): R_13,
        (0, 3): R_14,
        (1, 2): R_23,
        (1, 3): R_24,
        (2, 3): R_34
    }

    connections = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    connections = connections[:m]
    conductance_coeffs = [1/R_12, 1/R_13, 1/R_14, 1/R_23, 1/R_24, 1/R_34, 1/(2*R_env), 0]
    conductance_coeffs = conductance_coeffs[:m] + [1/(2*R_env)] + [0.0]*(int(2**np.ceil(np.log2(m+1))-m-1))
    C_l = np.sum(conductance_coeffs)**(-1/2)
    conductance_coeffs_amps = np.sqrt(conductance_coeffs)*C_l
    B = np.array([Q_1, Q_2, Q_3, Q_4])
    C_B = np.sum([el**2 for el in B])**(-1/2)
    B_amps = C_B*B

    kappa = 1
    pcoefs, C_p = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True, epsilon=epsilon)
    phi_pyqsp = QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx")
    C_p = C_p[0]


    q_aux = ["q"]
    l_aux = [f"l({i})" for i in range(int(np.log2(len(conductance_coeffs_amps))))]
    flag_aux = ["flag"]
    l_prime_aux = ["l'"]
    data_wires = [f"d({i})" for i in range(2)]

    wires = q_aux + l_aux + flag_aux + l_prime_aux + data_wires
    dev = qml.device("default.qubit", wires=wires)

    allowed_gates = {qml.CNOT, qml.RX, qml.RY, qml.RZ} 

    @partial(decompose, gate_set=allowed_gates)
    @qml.qnode(dev)
    def qsvt_circuit_demo(config):
        L_demo(B_amps, connections, conductance_coeffs_amps, phi_pyqsp, q_aux[0], config, l_aux, flag_aux[0], l_prime_aux[0], data_wires)
        return qml.state()
    start = time.time()
    specs_dict = qml.specs(qsvt_circuit_demo)(config=full_config[:m])
    end = time.time()
    print(f"m={m} done! ({end-start} seconds)")
    specs_dict['execution_time'] = end - start

    return specs_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, required=True)
    args = parser.parse_args()

    specs_dict = extract_decomposed_circuit(args.m, args.transpilation_option)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(script_dir, 'data', f"specs_dict_{args.m}_pl.pkl")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'wb') as f:
        pickle.dump(specs_dict, f)


if __name__ == '__main__':
    main()
