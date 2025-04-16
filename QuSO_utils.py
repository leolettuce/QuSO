import pennylane as qml
from pennylane import numpy as pnp
from matplotlib import pyplot as plt
import numpy as np
import math
import sys
import pyqsp
from pyqsp.angle_sequence import Polynomial, QuantumSignalProcessingPhases
import time
from sympy import symbols, sin, pi, solve, simplify
from functools import reduce
from operator import mul

def dec2bin(num, include_fraction=True, fraction_limit=10, bits=None):
    """Converts a decimal number to its binary representation.

    Args:
        num (float): Decimal number.
        include_fraction (boolean): If binary fractions should be used.
        fraction_limit (int): Fraction limit.
        bits (int): Number of bits.

    Returns:
        str: Binary number.
    """
    
    if num == 0:
        return "0".ljust(bits, '0') if bits is not None else "0"

    # Handle negative numbers
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    # Separate the integer and fractional parts of the number
    integer_part = int(num)
    fractional_part = num - integer_part

    # Convert integer part to binary
    binary_integer = bin(integer_part)[2:]  # Use bin() and strip the '0b' prefix

    # Convert fractional part to binary
    binary_fraction = ""
    if include_fraction and fractional_part > 0:
        binary_fraction += "."
        while fractional_part > 0 and len(binary_fraction) <= fraction_limit:
            fractional_part *= 2
            fractional_bit = int(fractional_part)
            binary_fraction += str(fractional_bit)
            fractional_part -= fractional_bit

    # Combine integer and fractional parts
    binary_result = binary_integer + binary_fraction

    # Adjust the result to match the specified number of bits if given
    if bits is not None:
        if include_fraction and '.' in binary_result:
            # Split into integer and fraction parts
            integer_part, fractional_part = binary_result.split('.')
            max_integer_bits = bits - len(fractional_part) - 1  # Account for the decimal point
            integer_part = integer_part.rjust(max_integer_bits, '0')[:max_integer_bits]
            fractional_part = fractional_part.ljust(bits - len(integer_part) - 1, '0')[:bits - len(integer_part) - 1]
            binary_result = integer_part + '.' + fractional_part
        else:
            binary_result = binary_result.rjust(bits, '0')[:bits]

    return sign + binary_result

def bin2dec(binary_str, twos_complement=False):
    """Converts a binary number to its decimal representation.

    Args:
        binary_str (str): Binary number as string.
        twos_complement (boolean): If twos complement should be used.

    Returns:
        float: Decimal number.
    """
    
    # Split the binary string into its integer and fractional parts
    integer_part, _, fraction_part = binary_str.partition('.')
    
    # Determine if the number is negative under two's complement rules
    is_negative = False
    if twos_complement:
        if integer_part:
            # Check if the integer part starts with a 1
            if integer_part[0] == '1':
                is_negative = True
        elif fraction_part and fraction_part[0] == '1':
            # No integer part, but fractional part starts with a 1
            is_negative = True
    
    # Handle two's complement for the integer part if needed
    if is_negative:
        # For integer part
        if integer_part:
            integer_value = int(integer_part, 2)
            integer_decimal = integer_value - (1 << len(integer_part))
        else:
            integer_decimal = 0
        
        # For fractional part, adjust to account for the two's complement
        fraction_value = sum(int(digit) * (2 ** -i) for i, digit in enumerate(fraction_part, start=1))
        fraction_decimal = fraction_value - (2 ** -len(fraction_part))
    else:
        # Convert integer part to decimal
        integer_decimal = int(integer_part, 2) if integer_part else 0

        # Convert fraction part to decimal
        fraction_decimal = sum(int(digit) * (2 ** -i) for i, digit in enumerate(fraction_part, start=1))
    
    # Combine both decimal values
    decimal_number = integer_decimal + fraction_decimal
    
    return decimal_number

def ctrl_op_list(op_list, control, control_values):
    """Controls the operations in op_list on the qubits according to control and control_values.

    Args:
        op_list (list): List of pennylane operations.
        control (wires): Control wires.
        control_values (list): Control bitstrings.

    Returns:
        list: List of controlled operations.
    """
    
    ops = []
    for op in op_list:
        ops.append(qml.ctrl(op, control=control, control_values=control_values))
    
    return ops

def block_encode_2x2_list(l_prime_aux, data):
    """Returns the list of operations for the 2x2 block encoding: 1-X.

    Args:
        l_prime_aux (wire): Auxiliary qubit for LCU block encoding.
        data (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []

    # prepare
    ops.append(qml.Hadamard(l_prime_aux))

    # select
    # ops.append(qml.ctrl(-qml.X(wires=data), control=l_prime_aux, control_values=1))
    ops.append(qml.ctrl(qml.GlobalPhase(np.pi, wires=data), control=l_prime_aux, control_values=1))
    ops.append(qml.ctrl(qml.X(wires=data), control=l_prime_aux, control_values=1))

    # unprepare
    ops.append(qml.Hadamard(l_prime_aux))

    return ops

def block_encode_2x2_first_qubit_list(flag_aux, l_prime_aux, d):
    """Returns the list of operations for the 2x2 block encoding in the upper left block of a larger matrix padded with zeros.

    Args:
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    lsb = d[-1]
    other = d[:-1]

    ops.append(qml.PauliX(wires=flag_aux))
    ops.append(qml.ctrl(qml.PauliX(wires=flag_aux), control=other, control_values=[0]*len(other)))
    
    ops += block_encode_2x2_list(l_prime_aux, lsb)

    return ops


def permute_block_list(i, j, d):
    """Returns the list of operations for permutation of i->0 and j->1 in the qubit register d.

    Args:
        i (int): Index for i-th row/column.
        j (int): Index for j-th row/column.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """

    def get_bit(number, index, bit_width):
        # Calculate the position of the bit in little-endian convention
        little_endian_index = bit_width - 1 - index
        return (number >> little_endian_index) & 1

    ops = []
    bit_width = len(d)
    
    # move the i state to the 0 state
    for k in range(bit_width):
        if get_bit(i, k, bit_width) == 1:
            ops.append(qml.PauliX(wires=d[k]))
    
    # get the 1st index for which j^i is not 0
    j_updated = j ^ i
    highest_nonzero_bit = int(math.log(j_updated & ((~j_updated) + 1), 2))
    highest_nonzero_bit_big_endian = bit_width - 1 - highest_nonzero_bit

    # flip all 1 bits in updated j conditioned on the 1st bit
    for k in range(bit_width):
        if k != highest_nonzero_bit_big_endian and get_bit(j_updated, k, bit_width):
            ops.append(qml.CNOT(wires=[d[highest_nonzero_bit_big_endian], d[k]]))

    # swap the qubit and the last qubit
    if highest_nonzero_bit_big_endian != bit_width-1:
        ops.append(qml.SWAP(wires=[d[bit_width-1], d[highest_nonzero_bit_big_endian]]))
    
    return ops

def block_encode_2x2_arbitrary_list(i, j, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the 2x2 block encoding at arbitrary positions i and j of a larger matrix padded with zeros.

    Args:
        i (int): Index for i-th row/column.
        j (int): Index for j-th row/column.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    permute_block_ops = permute_block_list(i, j, d)

    ops += permute_block_ops
    ops += block_encode_2x2_first_qubit_list(flag_aux, l_prime_aux, d)
    ops += [qml.adjoint(operation) for operation in permute_block_ops[::-1]]

    return ops

def combine_blocks_list(pair_list, coeffs, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the LCU block encoding of several 2x2 matrices placed in a larger matrix padded with zeros.

    Args:
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs (list): List of coefficients for every matrix.
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    # prepare
    ops.append(qml.StatePrep(coeffs, l_aux))

    # create list of control conditions
    ctrl_list = []
    for i in range(len(coeffs)):
        # Get the binary representation of i, remove the '0b' prefix, and pad with leading zeros
        bits = format(i, f'0{(len(coeffs)-1).bit_length()}b')
        # Convert the string of bits into a tuple of integers
        bits_tuple = tuple(int(bit) for bit in bits)
        ctrl_list.append(bits_tuple)

    # select
    for i, pair in enumerate(pair_list):
        ops += ctrl_op_list(block_encode_2x2_arbitrary_list(pair[0], pair[1], flag_aux, l_prime_aux, d), control=l_aux, control_values=ctrl_list[i])

    # unprepare  
    ops.append(qml.adjoint(qml.StatePrep(coeffs, l_aux)))

    return ops

def conditional_single_block_demo_list(i, j, condition_qubit, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the 2x2 block encoding at arbitrary positions i and j of a larger matrix padded with zeros 
    conditioned on condition_qubit.

    Args:
        i (int): Index for i-th row/column.
        j (int): Index for j-th row/column.
        condition_qubit (int): Either 0 or 1.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    if condition_qubit == 1:
        ops += block_encode_2x2_arbitrary_list(i, j, flag_aux, l_prime_aux, d)
    else:
        ops.append(qml.X(wires=flag_aux))
    # ops += ctrl_op_list(block_encode_2x2_arbitrary_list(i, j, flag_aux, l_prime_aux, d), control=condition_qubit, control_values=1)
    # ops.append(qml.ctrl(qml.X(wires=flag_aux), control=condition_qubit, control_values=0))

    return ops

def conditional_combine_blocks_demo_list(pair_list, coeffs, config, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the LCU block encoding of several 2x2 matrices placed in a larger matrix padded with zeros 
    conditioned on config.

    Args:
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs (list): List of coefficients for every matrix.
        config (list): List containing either 0 or 1.
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    # prepare
    ops.append(qml.StatePrep(coeffs, l_aux))

    # create list of control conditions
    ctrl_list = []
    for i in range(len(coeffs)):
        # Get the binary representation of i, remove the '0b' prefix, and pad with leading zeros
        bits = format(i, f'0{(len(coeffs)-1).bit_length()}b')
        # Convert the string of bits into a tuple of integers
        bits_tuple = tuple(int(bit) for bit in bits)
        ctrl_list.append(bits_tuple)
    # select
    for i, pair in enumerate(pair_list):
        ops += ctrl_op_list(conditional_single_block_demo_list(pair[0], pair[1], config[i], flag_aux, l_prime_aux, d), control=l_aux, control_values=ctrl_list[i])

    # unprepare
    ops.append(qml.adjoint(qml.StatePrep(coeffs, l_aux)))

    return ops

def conditional_single_block_list(i, j, condition_qubit, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the 2x2 block encoding at arbitrary positions i and j of a larger matrix padded with zeros 
    conditioned on condition_qubit.

    Args:
        i (int): Index for i-th row/column.
        j (int): Index for j-th row/column.
        condition_qubit (wire): Condition qubit.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    ops += ctrl_op_list(block_encode_2x2_arbitrary_list(i, j, flag_aux, l_prime_aux, d), control=condition_qubit, control_values=1)
    ops.append(qml.ctrl(qml.X(wires=flag_aux), control=condition_qubit, control_values=0))

    return ops

def conditional_combine_blocks_list(pair_list, coeffs, config, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the LCU block encoding of several 2x2 matrices placed in a larger matrix padded with zeros 
    conditioned on config.

    Args:
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs (list): List of coefficients for every matrix.
        config (wires): Condition qubits.
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    # prepare
    ops.append(qml.StatePrep(coeffs, l_aux))

    # create list of control conditions
    ctrl_list = []
    for i in range(len(coeffs)):
        # Get the binary representation of i, remove the '0b' prefix, and pad with leading zeros
        bits = format(i, f'0{(len(coeffs)-1).bit_length()}b')
        # Convert the string of bits into a tuple of integers
        bits_tuple = tuple(int(bit) for bit in bits)
        ctrl_list.append(bits_tuple)
    # select
    for i, pair in enumerate(pair_list):
        ops += ctrl_op_list(conditional_single_block_list(pair[0], pair[1], config[i], flag_aux, l_prime_aux, d), control=l_aux, control_values=ctrl_list[i])

    # unprepare
    ops.append(qml.adjoint(qml.StatePrep(coeffs, l_aux)))

    return ops

class CoolingSystemMatrix_demo(qml.operation.Operation):
    """Quantum operation for the block encoding of the Cooling System Matrix.

    The demo implementation requires only a list of conditional integers.
    """
    
    def __init__(self, pair_list, coeffs_list, config, l_aux, flag_aux, l_prime_aux, d):
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d
        super().__init__(wires=qml.wires.Wires(l_aux+[flag_aux]+[l_prime_aux]+d))

    def decomposition(self):
        return conditional_combine_blocks_demo_list(self.pair_list, self.coeffs_list, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d)
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=self.l_aux+[self.flag_aux]+[self.l_prime_aux]+self.d)
        return mat

class CoolingSystemMatrix(qml.operation.Operation):
    """Quantum operation for the block encoding of the Cooling System Matrix.

    The proper implementation requires qubits containing conditional variables.
    """
    
    def __init__(self, pair_list, coeffs_list, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=config+l_aux+[flag_aux]+[l_prime_aux]+d)
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        return conditional_combine_blocks_list(self.pair_list, self.coeffs_list, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d)
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=self.l_aux+[self.flag_aux]+[self.l_prime_aux]+self.d)
        return mat

def build_A(x, R_env, R_dict, connections):
    """Construction of matrix A conditioned on x.

    Args:
        x (list): List of conditional integers.

    Returns:
        array: Matrix A.
    """
    
    cons = []
    for i, x_ij in enumerate(x):
        if x_ij == 1:
            cons.append(connections[i])
    A = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                A[i, j] = 1/R_env + np.sum([1/R_dict[con] if i in con else 0 for con in cons])
            elif i < j:
                if (i, j) in cons:
                    A[i, j] = -1/R_dict[(i, j)]
            elif i > j:
                if (j, i) in cons:
                    A[i, j] = -1/R_dict[(j, i)]
    return A

def _qsp_to_qsvt(angles):
    r"""Converts qsp angles to qsvt angles."""
    num_angles = len(angles)
    update_vals = np.empty(num_angles)

    update_vals[0] = 3 * np.pi / 4
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = -np.pi / 4
    update_vals = qml.math.convert_like(update_vals, angles)

    return angles + update_vals

def custom_qsvt(blockencoding, angles, wires):
    r"""Returns custom QSVT operation."""
    
    angles = _qsp_to_qsvt(angles)
    global_phase = (len(angles) - 1) % 4
    projectors = []
    if global_phase:
        global_phase_op = qml.GlobalPhase(-0.5 * np.pi * (4 - global_phase), wires=wires)

    for idx, phi in enumerate(angles):
        projectors.append(qml.PCPhase(phi, dim=4, wires=wires))

    projectors = projectors[::-1]  # reverse order to match equation

    if global_phase:
        return qml.prod(global_phase_op, qml.QSVT(blockencoding, projectors))
    return qml.QSVT(blockencoding, projectors)

class L_demo(qml.operation.Operation):
    """Quantum operation for the linear system solver.

    The demo implementation requires only a list of conditional integers.
    """
    
    def __init__(self, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=[q_aux]+l_aux+[flag_aux, l_prime_aux]+d)
        self.b = b
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.angles = angles
        self.q_aux = q_aux
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        ops = []
        ops.append(qml.StatePrep(self.b, wires=self.d))
        ops.append(qml.Hadamard(wires=self.q_aux))
        ops.append(qml.ctrl(custom_qsvt(
            CoolingSystemMatrix_demo(self.pair_list, self.coeffs_list, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d),
            self.angles, 
            self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d), control=self.q_aux, control_values=(0,)))
        ops.append(qml.ctrl(qml.adjoint(custom_qsvt(CoolingSystemMatrix_demo(self.pair_list, self.coeffs_list, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d), self.angles, self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)), control=self.q_aux, control_values=(1,)))
        ops.append(qml.Hadamard(wires=self.q_aux))

        return ops
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=[self.q_aux]+self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)
        return mat

class L(qml.operation.Operation):
    """Quantum operation for the linear system solver.

    The proper implementation requires qubits containing conditional variables.
    """
    
    def __init__(self, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=config+[q_aux]+l_aux+[flag_aux, l_prime_aux]+d)
        self.b = b
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.angles = angles
        self.q_aux = q_aux
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        ops = []
        ops.append(qml.StatePrep(self.b, wires=self.d))
        ops.append(qml.Hadamard(wires=self.q_aux))
        ops.append(qml.ctrl(custom_qsvt(CoolingSystemMatrix(self.pair_list, self.coeffs_list, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d), self.angles, self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d), control=self.q_aux, control_values=(0,)))
        ops.append(qml.ctrl(qml.adjoint(custom_qsvt(CoolingSystemMatrix(self.pair_list, self.coeffs_list, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d), self.angles, self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)), control=self.q_aux, control_values=(1,)))
        ops.append(qml.Hadamard(wires=self.q_aux))

        return ops
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=self.config+[self.q_aux]+self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)
        return mat
    
def grover_operator_demo_list(psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the Grover operator.

    Args:
        psi (list): List containing integers corresponding to the bitstring corresponding to the state of interest.
        b (list): B vector.
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs_list (list): List of coefficients for every matrix.
        angles (array): Array containing phase factors for QSVT.
        q_aux (wire): Auxilliary qubit for QSVT.
        config (list): List containing conditional integers (Either 0 or 1).
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    wires = [q_aux]+l_aux+[flag_aux, l_prime_aux]+d
    ops.append(qml.FlipSign([0]*(len(l_aux)+3) + psi, wires=wires))    # S_psi
    ops.append(qml.adjoint(L_demo(b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d)))                    # A^dagger
    ops.append(qml.FlipSign([0]*(len(l_aux)+3+len(d)), wires=wires))    # S_0
    ops.append(L_demo(b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d))                                 # A
    ops.append(qml.GlobalPhase(np.pi, wires=wires))                             # -1

    return ops

class grover_operator_demo(qml.operation.Operation):
    """Quantum operation for the Grover operator.

    The demo implementation requires only a list of conditional integers.
    """
    
    def __init__(self, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=[q_aux]+l_aux+[flag_aux, l_prime_aux]+d)
        self.psi = psi
        self.b = b
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.angles = angles
        self.q_aux = q_aux
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        return grover_operator_demo_list(self.psi, self.b, self.pair_list, self.coeffs_list, self.angles, self.q_aux, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d)
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=[self.q_aux]+self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)
        return mat

def grover_operator_list(psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for the Grover operator.

    Args:
        psi (list): List containing integers corresponding to the bitstring corresponding to the state of interest.
        b (list): B vector.
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs_list (list): List of coefficients for every matrix.
        angles (array): Array containing phase factors for QSVT.
        q_aux (wire): Auxilliary qubit for QSVT.
        config (wires): Condition qubits.
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    wires = [q_aux]+l_aux+[flag_aux, l_prime_aux]+d
    ops.append(qml.FlipSign([0]*(len(l_aux)+3) + psi, wires=wires))    # S_psi
    ops.append(qml.adjoint(L(b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d)))                    # A^dagger
    ops.append(qml.FlipSign([0]*(len(l_aux)+3+len(d)), wires=wires))    # S_0
    ops.append(L(b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d))                                 # A
    ops.append(qml.GlobalPhase(np.pi, wires=wires))                             # -1

    return ops

class grover_operator(qml.operation.Operation):
    """Quantum operation for the Grover Operator.

    The proper implementation requires qubits containing conditional variables.
    """
    
    def __init__(self, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=config+[q_aux]+l_aux+[flag_aux, l_prime_aux]+d)
        self.psi = psi
        self.b = b
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.angles = angles
        self.q_aux = q_aux
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        return grover_operator_list(self.psi, self.b, self.pair_list, self.coeffs_list, self.angles, self.q_aux, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d)

    @staticmethod
    def compute_decomposition(psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        grover_operator_list(psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d)
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=self.config+[self.q_aux]+self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)
        return mat

def QAE_demo_list(phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for QAE.

    Args:
        phase (wires): Phase qubits.
        psi (list): List containing integers corresponding to the bitstring corresponding to the state of interest.
        b (list): B vector.
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs_list (list): List of coefficients for every matrix.
        angles (array): Array containing phase factors for QSVT.
        q_aux (wire): Auxilliary qubit for QSVT.
        config (list): List containing conditional integers (Either 0 or 1).
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    ops.append(L_demo(b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d))
    for wire in phase:
        ops.append(qml.Hadamard(wires=wire))
    ops.append(qml.ControlledSequence(grover_operator_demo(psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d), control=phase))
    ops.append(qml.adjoint(qml.QFT)(wires=phase))       

    return ops

class QAE_demo(qml.operation.Operation):
    """Quantum operation for QAE.

    The demo implementation requires only a list of conditional integers.
    """
    
    def __init__(self, phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=phase+[q_aux]+l_aux+[flag_aux, l_prime_aux]+d)
        self.phase = phase
        self.psi = psi
        self.b = b
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.angles = angles
        self.q_aux = q_aux
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        return QAE_demo_list(self.phase, self.psi, self.b, self.pair_list, self.coeffs_list, self.angles, self.q_aux, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d)
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=self.phase+[self.q_aux]+self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)
        return mat

def QAE_list(phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
    """Returns the list of operations for QAE.

    Args:
        phase (wires): Phase qubits.
        psi (list): List containing integers corresponding to the bitstring corresponding to the state of interest.
        b (list): B vector.
        pair_list (list): List of indices for i-th and j-th row/column.
        coeffs_list (list): List of coefficients for every matrix.
        angles (array): Array containing phase factors for QSVT.
        q_aux (wire): Auxilliary qubit for QSVT.
        config (wires): Conditional wires.
        l_aux (wires): Auxiliary qubits for outer LCU block encoding.
        flag_aux (wire): Flag qubit for flagging rows/columns with zero entries.
        l_prime_aux (wire): Auxiliary qubit for inner LCU block encoding.
        d (wires): Data qubits.

    Returns:
        list: List of operations.
    """
    
    ops = []
    ops.append(L(b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d))
    for wire in phase:
        ops.append(qml.Hadamard(wires=wire))
    ops.append(qml.ControlledSequence(grover_operator(psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d), control=phase))
    ops.append(qml.adjoint(qml.QFT)(wires=phase))       

    return ops

class QAE(qml.operation.Operation):
    """Quantum operation for QAE.

    The proper implementation requires qubits containing conditional variables.
    """
    
    def __init__(self, phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
        super().__init__(wires=config+phase+[q_aux]+l_aux+[flag_aux, l_prime_aux]+d)
        self.phase = phase
        self.psi = psi
        self.b = b
        self.pair_list = pair_list
        self.coeffs_list = coeffs_list
        self.angles = angles
        self.q_aux = q_aux
        self.config = config
        self.l_aux = l_aux
        self.flag_aux = flag_aux
        self.l_prime_aux = l_prime_aux
        self.d = d

    def decomposition(self):
        return QAE_list(self.phase, self.psi, self.b, self.pair_list, self.coeffs_list, self.angles, self.q_aux, self.config, self.l_aux, self.flag_aux, self.l_prime_aux, self.d)
    
    def expand(self):
        return qml.tape.QuantumTape(self.decomposition())
    
    def compute_matrix(self):
        op_list = self.decomposition()
        mat = qml.matrix(qml.prod(*tuple(op_list[::-1])), wire_order=self.config+self.phase+[self.q_aux]+self.l_aux+[self.flag_aux, self.l_prime_aux]+self.d)
        return mat

def polynomial_to_hamiltonian(c, polynomial, phase_wires):
    """Returns the Hamiltonian for a given polynomial.

    Args:
        c (float): Constant scaling factor.
        polynomial (dict): Dictionary containing the polynomial coefficients.
        phase_wires (wires): Phase qubits.

    Returns:
        operation: Hamiltonian.
    """
    
    coeffs = []
    obs = []
    for var_list, coeff in polynomial.items():
        if np.abs(coeff) > 0:
            pauli_string = qml.PauliZ(wires=phase_wires[0]) if var_list[0] else qml.Identity(wires=phase_wires[0])
            for i, bit in enumerate(var_list[1:]):
                pauli_string = pauli_string @ qml.PauliZ(wires=phase_wires[i+1]) if bit else pauli_string @ qml.Identity(wires=phase_wires[i+1])
            obs.append(pauli_string)
            coeffs.append(c * coeff)

    return qml.Hamiltonian(coeffs, obs)

def polynomial_coefficients_from_function(func, num_bits):
    """Returns the polynomial coefficients for a function approximation.

    Args:
        func (Callable): Function that should be approximated.
        num_bits (int): Number of qubits.

    Returns:
        dict: Polynomial.
    """
    
    MIN_COEFF = 10e-7

    def prod(iterable):
        return reduce(mul, iterable, 1)

    lookup_table = {tuple(map(int, format(x, f'0{num_bits}b'))): func(x / (2**num_bits - 1) - 1) for x in range(2 ** num_bits)}

    x_symbols = symbols(f'x0:{num_bits}')
    a_symbols = symbols(f'a0:{2 ** num_bits}')

    # Construct the polynomial expression
    polynomial_expr = sum(a_symbols[i] * prod(
        (x_symbols[j]) ** int(bit) for j, bit in enumerate(f'{i:0{num_bits}b}')) for i
                          in range(2 ** num_bits))

    # Setup equations based on the lookup table
    equations = [polynomial_expr.subs(dict(zip(x_symbols, k))) - v for k, v in
                 lookup_table.items()]

    # Solve for coefficients
    solved_coeffs = solve(equations, a_symbols)

    # Clean small coefficients
    solved_coeffs = {var: float(coeff) if abs(coeff) > MIN_COEFF else 0 for var, coeff in solved_coeffs.items()}

     # Substitute the solved coefficients back into the polynomial expression
    polynomial_with_coeffs = polynomial_expr.subs(solved_coeffs)
    print("Polynomial Expression:", polynomial_with_coeffs)
    
    # Create substitution dictionary for 1 - x_i/2
    substitution_dict = {x: (1 - x) / 2 for x in x_symbols}

    # Apply the substitution to the polynomial expression
    polynomial_with_substitution = polynomial_with_coeffs.subs(substitution_dict)

    # Simplify the polynomial expression after substitution
    simplified_polynomial = simplify(polynomial_with_substitution)

    # Extract coefficients after substitution
    substituted_coefficients_dict = {}
    for i in range(2 ** num_bits):
        binary_tuple = tuple(int(bit) for bit in format(i, f'0{num_bits}b'))
        term = prod(x_symbols[j] ** int(bit) for j, bit in enumerate(binary_tuple))
        coefficient = simplified_polynomial.as_coefficients_dict().get(term, 0)
        substituted_coefficients_dict[binary_tuple] = float(coefficient) if abs(coefficient) > MIN_COEFF else 0

    # Handle the constant term
    constant_term = simplified_polynomial.as_coefficients_dict().get(1, 0)
    if abs(constant_term) > MIN_COEFF:
        substituted_coefficients_dict[(0,) * num_bits] = float(constant_term)

    return substituted_coefficients_dict, simplified_polynomial


def get_func_hamiltonian(func, c, phase_wires):
    """Returns the Hamiltonian for a given function.

    Args:
        func (Callable): Function that should be approximated.
        c (float): Constant scaling factor.
        phase_wires (wires): Phase qubits.

    Returns:
        operation: Hamiltonian.
    """
    
    # Compute the coefficients and polynomial
    coefficients_dict, polynomial = polynomial_coefficients_from_function(func, len(phase_wires))
    
    # can be done exact with polynomial of the same order of the function
    return polynomial_to_hamiltonian(c, coefficients_dict, phase_wires)

def qae_func(theta):
    return np.sin(np.pi*theta)

def phase_application(gamma, hamiltonian):
    """Applies exp(i*gamma*func(theta)).

    Args:
        gamma (float): Variational parameter for QAOA.
        hamiltonian (operation): Hamiltonian approximating the desired function.
    """
    
    qml.templates.ApproxTimeEvolution(hamiltonian, gamma, 1)

def cost_layer(gamma, hamiltonian, phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d):
    QAE(phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d)
    phase_application(gamma, hamiltonian)
    qml.adjoint(QAE(phase, psi, b, pair_list, coeffs_list, angles, q_aux, config, l_aux, flag_aux, l_prime_aux, d))
