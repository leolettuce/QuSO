import numpy as np
from classiq import *
from classiq.execution import ExecutionPreferences
from classiq.open_library.functions import hadamard_transform
import pyqsp
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from classiq.qmod.symbolic import log, logical_and
from classiq.execution import ExecutionPreferences, ClassiqSimulatorBackendNames, ClassiqBackendPreferences
import pickle 
import time
import argparse
import os

@qfunc
def block_encode_2x2(aux: QNum, data: QArray[QBit]):
    """
    Block encoding for 2x2 matrix
    """
    within_apply(
        lambda: H(aux),
        lambda: control(aux == 1, lambda: (X(data), U(0, 0, 0, np.pi, data))),
    )

@qfunc
def block_encode_2X2_first_qubit(flag: QBit, aux: QBit, data: QArray[QBit]):
    lsb = QBit("lsb")
    msb = QNum("msb", data.len - 1, False, 0)

    bind(data, [lsb, msb])
    flag ^= msb > 0

    block_encode_2x2(aux, lsb)
    bind([lsb, msb], data)

@qfunc
def block_encode_2X2_arbitrary(
    i: CInt, j: CInt, flag: QBit, aux: QBit, data: QArray[QBit]
):
    within_apply(
        lambda: permute_block(i, j, data),
        lambda: block_encode_2X2_first_qubit(flag, aux, data),
    )


@qfunc
def permute_block(i: CInt, j: CInt, data: QArray[QBit]):
    def get_bit(number, index):
        return (number >> index) & 1

    # move the i state to the 0 state
    repeat(
        data.len,
        lambda k: if_(
            get_bit(i, k) == 1, lambda: X(data[k]), lambda: IDENTITY(data[k])
        ),
    )

    #  # get the 1st index for which j^i is not 0
    j_updated = j ^ i
    highest_nonzero_bit = log(j_updated & ((~j_updated) + 1), 2)

    # # filp all 1 bits in updated j conditioned on the 1st bit
    repeat(
        data.len,
        lambda k: if_(
            logical_and(k != highest_nonzero_bit, get_bit(j_updated, k)),
            lambda: CX(data[highest_nonzero_bit], data[k]),
            lambda: IDENTITY(data),
        ),
    )

    # swap the qbit and the 0 qbit
    if_(highest_nonzero_bit != 0, lambda: SWAP(data[0], data[highest_nonzero_bit]), lambda: IDENTITY(data))

@qfunc
def combine_blocks(
    pair_list: CArray[CArray[CInt]],
    lcu_aux: QNum,
    flag: QBit,
    aux: QBit,
    data: QArray[QBit],
):
    """
    expecting list of distinct pairs
    also assuming at the moment that the size if a power of 2 (for the state preparation)
    """
    within_apply(
        lambda: hadamard_transform(lcu_aux),
        lambda: repeat(
            pair_list.len,
            lambda index: control(
                lcu_aux == index,
                lambda: block_encode_2X2_arbitrary(
                    pair_list[index][0], pair_list[index][1], flag, aux, data
                ),
            ),
        ),
    )

@qfunc
def combine_blocks_coeffs(
    pair_list: CArray[CArray[CInt]],
    amplitudes: CArray[CReal],
    lcu_aux: QNum,
    flag: QBit,
    aux: QBit,
    data: QArray[QBit],
):
    """
    expecting list of distinct pairs
    also assuming at the moment that the size if a power of 2 (for the state preparation)
    """
    within_apply(
        lambda: inplace_prepare_amplitudes(amplitudes, 0, lcu_aux),
        lambda: repeat(
            pair_list.len,
            lambda index: control(
                lcu_aux == index,
                lambda: block_encode_2X2_arbitrary(
                    pair_list[index][0], pair_list[index][1], flag, aux, data
                ),
            ),
        ),
    )

@qfunc
def conditional_single_block(
    i: CInt, j: CInt, condition_var: QNum, flag: QBit, aux: QBit, data: QArray[QBit]
):
    """
    assuming to be in the context of LCU, conditioned on specific index
    """
    control(
        condition_var == 1, lambda: block_encode_2X2_arbitrary(i, j, flag, aux, data)
    )
    # else set flag to get 0 matrix
    control(condition_var == 0, lambda: X(flag))


@qfunc
def conditional_combine_blocks(
    pair_list: CArray[CArray[CInt]],
    u: QArray[QBit],
    lcu_aux: QNum,
    flag: QBit,
    aux: QBit,
    data: QArray[QBit],
):
    within_apply(
        lambda: hadamard_transform(lcu_aux),
        lambda: repeat(
            pair_list.len,
            lambda index: control(
                lcu_aux == index,
                lambda: conditional_single_block(
                    pair_list[index][0], pair_list[index][1], u[index], flag, aux, data
                ),
            ),
        ),
    )

@qfunc
def conditional_single_block(
    i: CInt, j: CInt, condition_var: QNum, flag: QBit, aux: QBit, data: QArray[QBit]
):
    """
    assuming to be in the context of LCU, conditioned on specific index
    """
    control(
        condition_var == 1, lambda: block_encode_2X2_arbitrary(i, j, flag, aux, data)
    )
    # else set flag to get 0 matrix
    control(condition_var == 0, lambda: X(flag))


@qfunc
def conditional_combine_blocks_coeffs(
    pair_list: CArray[CArray[CInt]],
    amplitudes: CArray[CReal],
    u: QArray[QBit],
    lcu_aux: QNum,
    flag: QBit,
    aux: QBit,
    data: QArray[QBit],
):
    within_apply(
        lambda: inplace_prepare_amplitudes(amplitudes, 0, lcu_aux),
        lambda: repeat(
            pair_list.len,
            lambda index: control(
                lcu_aux == index,
                lambda: conditional_single_block(
                    pair_list[index][0], pair_list[index][1], u[index], flag, aux, data
                ),
            ),
        ),
    )

def extract_decomposed_circuit(m=1, transpilation_option="decompose"):
    epsilon=0.1
    kappa = 1
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

    lcu_aux_size = int(np.log2(len(conductance_coeffs_amps)))
    data_size = int(np.log2(len(B_amps)))

    def getOneOverXPhases(epsilon=0.05, kappa=5):
        pcoefs, C_p = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True, epsilon=epsilon)
        phi_pyqsp = QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx")
        C_p = C_p[0]

        # change the R(x) to W(x), as the phases are in the W(x) conventions
        phases = np.array(phi_pyqsp)
        phases[1:-1] = phases[1:-1] - np.pi / 2
        phases[0] = phases[0] - np.pi / 4
        # TODO: Check this, this comes out of the tutorial, in our implementation we had the following line instead:
        # phases[-1] = phases[-1] + np.pi / 4 + np.pi / 2
        phases[-1] = phases[-1] + (2 * (len(phases) - 1) - 1) * np.pi / 4

        # verify conventions. minus is due to exp(-i*phi*z) in qsvt in comparison to qsp
        phases = -2 * phases

        return phases, C_p

    class Block(QStruct):
        lcu_aux: QNum[lcu_aux_size, False, 0]
        flag: QBit
        aux: QBit 

    class BlockEncodedState(QStruct):
        block: Block
        data: QArray[QBit, data_size] 

    class QsvtState(QStruct):
        qsvt_aux: QBit # 1
        state: BlockEncodedState

    @qfunc
    def identify_block(state: BlockEncodedState, qbit: QBit):
        block_qubits = QNum("block_qubits", state.block.size, False, 0)
        data = QArray("data", length=state.data.size)

        bind(state, [block_qubits, data])
        qbit ^= block_qubits == 0
        bind([block_qubits, data], state)


    @qfunc(synthesize_separately=True)
    def qsvt_solve_system(
        b_amps: CArray[CReal],
        block_encoding: QCallable[QArray[QBit]],
        phases: CArray[CReal],
        qsvt_state: QsvtState,
    ) -> None:
        """QSVT implementation to solve a linear system Ax=b.

        Args:
            pair_list (CArray[CArray[CInt]]): _description_
            lcu_amps (CArray[CReal]): _description_
            b_amps (CArray[CReal]): Values of the b vector.
            u_var (QArray[QBit]): Decision variables which are controlled by the outer optimization loop and and control the construction of the matrix A.
            qsvt_state (QsvtState): _description_
        """

        # Prepare b as a quantum state in amplitude encoding.
        inplace_prepare_amplitudes(b_amps, 0, qsvt_state.state.data)

        qsvt_real_aux = QBit("qsvt_real_aux")
        allocate(1, qsvt_real_aux)
        hadamard_transform(qsvt_real_aux)
        control(
            qsvt_real_aux==0,
            lambda: qsvt_inversion(
                phase_seq=phases,#.tolist(),
                block_encoding_cnot=lambda qvar, qbit: identify_block(qvar, qbit),
                u=lambda qvar: block_encoding(qvar),
                qvar=qsvt_state.state,
                aux=qsvt_state.qsvt_aux,
            )
        )
        control(
            qsvt_real_aux==1,
            lambda: invert(
                lambda: qsvt_inversion(
                    phase_seq=phases,#.tolist(),
                    block_encoding_cnot=lambda qvar, qbit: identify_block(qvar, qbit),
                    u=lambda qvar: block_encoding(qvar),
                    qvar=qsvt_state.state,
                    aux=qsvt_state.qsvt_aux,
                )
            )
        )

        
        hadamard_transform(qsvt_real_aux)

    phases, C_p = getOneOverXPhases(epsilon=epsilon, kappa=kappa)

    @qfunc
    def block_encoding_demo(
        pair_list: CArray[CArray[CInt]],
        amplitudes: CArray[CReal],
        state: BlockEncodedState
    ):
        lcu_aux = state.block.lcu_aux
        flag = state.block.flag
        aux = state.block.aux
        data = state.data

        combine_blocks_coeffs(pair_list = pair_list, amplitudes = amplitudes, lcu_aux = lcu_aux, flag = flag, aux = aux, data = data)


    @qfunc
    def conditional_block_encoding(
        pair_list: CArray[CArray[CInt]],
        amplitudes: CArray[CReal],
        u: QArray[QBit],
        state: BlockEncodedState
    ):
        lcu_aux = state.block.lcu_aux
        flag = state.block.flag
        aux = state.block.aux
        data = state.data

        conditional_combine_blocks_coeffs(pair_list = pair_list, amplitudes = amplitudes, u = u, lcu_aux = lcu_aux, flag = flag, aux = aux, data = data)

    @qfunc
    def main(
        qsvt_state: Output[QsvtState]
    ): 
        allocate(qsvt_state)

        qsvt_solve_system(
            b_amps = B_amps,
            block_encoding = lambda q_var: block_encoding_demo(pair_list = connections, amplitudes = conductance_coeffs_amps, state = q_var),
            phases = phases,
            qsvt_state = qsvt_state
        )

    def get_qprog(main):
        execution_preferences = ExecutionPreferences(
            num_shots=None,
            backend_preferences=ClassiqBackendPreferences(
                backend_name=ClassiqSimulatorBackendNames.SIMULATOR_STATEVECTOR
            ),
        )
        custom_hardware_settings = CustomHardwareSettings(basis_gates=["cx", "rx", "ry", "rz"])
        synthesis_preferences = Preferences(
            transpilation_option = transpilation_option, 
            timeout_seconds = 14400,
            custom_hardware_settings=custom_hardware_settings
        )

        qmod = create_model(main, execution_preferences=execution_preferences, preferences=synthesis_preferences)
        qmod = set_execution_preferences(qmod, execution_preferences)
        qprog = synthesize(qmod)
        return qprog

    start = time.time()
    qprog=get_qprog(main)
    end = time.time()
    print(f"m={m}, transpilation_option={transpilation_option} done! ({end-start} seconds)")
    circuit = QuantumProgram.from_qprog(qprog)
    specs_dict = circuit.transpiled_circuit.count_ops
    specs_dict['depth'] = circuit.transpiled_circuit.depth
    specs_dict['execution_time'] = end - start

    return specs_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--transpilation_option', type=str, required=True)
    args = parser.parse_args()

    specs_dict = extract_decomposed_circuit(args.m, args.transpilation_option)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(script_dir, 'data', f"specs_dict_{args.m}_cq_{args.transpilation_option}.pkl")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, 'wb') as f:
        pickle.dump(specs_dict, f)


if __name__ == '__main__':
    main()

