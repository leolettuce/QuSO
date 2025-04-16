import numpy as np
import pennylane as qml

class qpa(qml.operation.Operation):
    def __init__(self, gamma, phase_wires, twos_complement=False, id=None):
        super().__init__(wires=phase_wires, id=id)
        self.gamma = gamma
        self.phase_wires = phase_wires
        self.twos_complement = twos_complement

    def decomposition(self):
        ops = []
        alphas = [2**(-i-1) for i in range(len(self.phase_wires))]
        if self.twos_complement:
            alphas[0] *= -1
        for i, wire in enumerate(self.phase_wires):
            ops.append(qml.PhaseShift(-self.gamma*alphas[i], wires=wire))
        return ops


def compute_u_g(k: int) -> np.ndarray:
    """Computes the U_g Matrix for a given QPA precision k.
       The U_g matrix represents the function g(z) = (1/2) * (2 * sin^2(pi * z) - 1) by mapping the tensor products
       of all possible bitstrings (of length k) to the corresponding tensor product of the bitstring (of length k)
       representing the function value g(z) in two's complement fixed point representation.
       I.e., the U_g matrix is a 2^k x 2^k matrix and all input and output vectors are of
       the form (0, 0, ..., 0, 1, 0, ..., 0)^T, containing exactly one 1 at any position.

    :param k: The QPA precision.
    :type k: int

    :returns: The U_g Matrix.
    """

    solution_vectors = []

    for i in range(2 ** k):
        bitstring = format(i, f'0{k}b')  # Create bitstring
        decimal_z = midpoint_twos_complement_to_decimal(bitstring)  # convert to decimal
        function_result = scaled_g_of_z(decimal_z)  # apply suitably scaled g function
        binary_fixed_point = decimal_to_midpoint_twos_complement(function_result, k)  # convert to binary fixed point
        solution_vectors.append(binary_string_to_tensor_product(binary_fixed_point))  # append to solution vectors

    return np.column_stack(solution_vectors)


def scaled_g_of_z(z: float) -> float:
    """Applies the function g(z) = (1/2) * sin(pi * z) to the input z.
       The function g(z) is scaled by a factor of 1/2.

    :param z: The input value for the function g(z).
    :type z: float

    :returns: The result of the function g(z).
    """

    return np.sin(np.pi * z) / 2


def midpoint_twos_complement_to_decimal(bitstring: str) -> float:
    """Converts a bitstring in two's complement fixed point representation to a decimal value, corresponding to
       the following formula:
            decimal = -0.5 * bitstring[0] + sum_{i=1}^{k} 2^(-(i+1)) * bitstring[i]

    :param bitstring: The bitstring in two's complement fixed point representation.
    :type bitstring: str

    :returns: The decimal value.
    """

    value = 0.0

    # First bit represents -0.5
    if bitstring[0] == '1':
        value -= 0.5

    for i in range(1, len(bitstring)):
        if bitstring[i] == '1':
            value += 2 ** -(i + 1)

    return value


def decimal_to_midpoint_twos_complement(value: float, bit_length: int) -> str:
    """Converts a decimal value to a bitstring in two's complement fixed point representation, corresponding to
       the following formula:
            decimal = -0.5 * bitstring[0] + sum_{i=1}^{k} 2^(-(i+1)) * bitstring[i]

    :param value: The decimal value.
    :type value: float

    :param bit_length: The length of the bitstring.
    :type bit_length: int

    :returns: The bitstring in two's complement fixed point representation.
    """

    if value == 0.5:
        return '0' + '1' * (bit_length - 1)

    if value < -0.5 or value >= 0.5:
        raise ValueError(f"Value has to be within interval [-0.5, 0.5) - Actual value: {value}")

    if value < 0:
        bitstring = '1'
        value += 0.5  # value now in [0, 0.5)
    else:
        bitstring = '0'

    for i in range(1, bit_length):
        bit_value = 2 ** -(i + 1)
        if value >= bit_value:
            bitstring += '1'
            value -= bit_value
        else:
            bitstring += '0'

    return bitstring


def binary_string_to_tensor_product(binary_string: str) -> np.ndarray:
    """Interprets the given binary_string as a list of qubits ('0' as |0> and '1' as |1>)
       and returns their tensor product.

    :param binary_string: The binary string.
    :type binary_string: str

    :returns: The tensor product of the corresponding bitstrings.
    """

    ket_0, ket_1 = np.array([1, 0]), np.array([0, 1])
    tensor_product = ket_0 if binary_string[0] == '0' else ket_1

    for bit in binary_string[1:]:
        if bit == '0':
            tensor_product = np.kron(tensor_product, ket_0)
        else:
            tensor_product = np.kron(tensor_product, ket_1)

    return tensor_product


def tensor_product_to_binary_string(tensor_product: np.ndarray) -> str:
    """Assumes the given tensor_product to be an arbitrary tensor product of the computational basis vectors |0> and |1>
       (i.e., |0> = [1, 0]^T and |1> = [0, 1]^T and thus, any tensor product of these vectors is a vector
       containing exactly one 1 at any position and 0 elsewhere). Converts the tensor product to the corresponding
       binary string representation (i.e., '0' for |0> and '1' for |1>).

    :param tensor_product: The tensor product of the corresponding bitstrings.
    :type tensor_product: np.ndarray

    :returns: The binary string.
    """

    assert (float(int(np.log2(len(tensor_product)))) == np.log2(len(tensor_product)))

    k = int(np.log2(len(tensor_product)))

    binary_string = ''

    for i in range(0, k):
        mid_index = len(tensor_product) // 2
        segment1 = tensor_product[:mid_index]
        segment2 = tensor_product[mid_index:]

        if 1 in segment1:
            binary_string += '0'
            tensor_product = segment1
        else:
            binary_string += '1'
            tensor_product = segment2

    return binary_string
