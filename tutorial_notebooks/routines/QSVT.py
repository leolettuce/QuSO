import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt


# QSVT with PennyLane polynomial (cf. https://pennylane.ai/qml/demos/tutorial_apply_qsvt/)

def sum_even_odd_circ_controlled_list(x, phi, ancilla_wire, wires, control, control_values):
    phi1, phi2 = phi[: len(phi) // 2], phi[len(phi) // 2:]

    ops = []

    ops.append(qml.ctrl(qml.Hadamard(wires=ancilla_wire), control=control, control_values=control_values))  # equal superposition

    # apply even and odd polynomial approx
    ops.append(qml.ctrl(qml.qsvt(x, phi1, wires=wires), control=(ancilla_wire, control), control_values=(0, control_values)))
    ops.append(qml.ctrl(qml.qsvt(x, phi2, wires=wires), control=(ancilla_wire, control), control_values=(1, control_values)))

    ops.append(qml.ctrl(qml.Hadamard(wires=ancilla_wire), control=control, control_values=control_values))  # un-prepare superposition

    return ops

def real_u_list(A, phi, qsvt_wires):
    ops = []

    ops.append(qml.Hadamard(wires=qsvt_wires[0]))

    ops += sum_even_odd_circ_controlled_list(A, phi, qsvt_wires[1], qsvt_wires[2:], control=qsvt_wires[0], control_values=0)
    ops += [qml.adjoint(operation) for operation in sum_even_odd_circ_controlled_list(A.T, phi, qsvt_wires[1], qsvt_wires[2:], control=qsvt_wires[0], control_values=1)]

    ops.append(qml.Hadamard(wires=qsvt_wires[0]))

    return ops

def qsvt_ls_list(A, b, phi, qsvt_wires):
    ops = []
    ops.append(qml.StatePrep(b, wires=qsvt_wires[3:]))
    ops += real_u_list(A.T, phi, qsvt_wires)  # invert the singular values of A transpose to get A^-1

    return ops

class qsvt_ls_PL(qml.operation.Operation):
    def __init__(self, A, b, phi, wires, id=None):
        super().__init__(wires=wires, id=id)
        self.A = A
        self.b = b
        self.phi = phi

    def decomposition(self):
        # Define the sequence of operations that make up the custom operation
        return qsvt_ls_list(self.A, self.b, self.phi, self.wires)

def sum_even_odd_circ(x, phi, ancilla_wire, wires):
    phi1, phi2 = phi[: len(phi) // 2], phi[len(phi) // 2:]

    qml.Hadamard(wires=ancilla_wire)  # equal superposition

    # apply even and odd polynomial approx
    qml.ctrl(qml.qsvt, control=(ancilla_wire,), control_values=(0,))(x, phi1, wires=wires)
    qml.ctrl(qml.qsvt, control=(ancilla_wire,), control_values=(1,))(x, phi2, wires=wires)

    qml.Hadamard(wires=ancilla_wire)  # un-prepare superposition

def plot_poly_PL(phi, s, kappa):
    x_vals = np.linspace(s, 1, 50)
    inv_x = [s * (1 / x) for x in x_vals]

    samples_x = np.linspace(0, 1, 100)
    qsvt_y_vals = [
        np.real(qml.matrix(sum_even_odd_circ, wire_order=["ancilla", 0])(x, phi, "ancilla", wires=[0])[0, 0])
        for x in samples_x
    ]

    plt.plot(samples_x, qsvt_y_vals, label="Re(qsvt)")
    plt.plot(x_vals, inv_x, label="target")

    plt.vlines(1 / kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
    plt.vlines(0.0, -1.0, 1.0, color="black")
    plt.hlines(0.0, -0.1, 1.0, color="black")

    plt.legend()
    plt.show()


# QSVT with pyqsp polynomial (cf. https://github.com/ichuang/pyqsp)

def real_pyqsp_list(A, phi, qsvt_wires):
    ops = []

    ops.append(qml.Hadamard(wires=qsvt_wires[0]))

    ops.append(qml.ctrl(qml.qsvt(A, phi, qsvt_wires[1:], convention="Wx"), control=qsvt_wires[0], control_values=(0,)))
    ops.append(qml.adjoint(qml.ctrl(qml.qsvt(A.T, phi, qsvt_wires[1:], convention="Wx"), control=qsvt_wires[0], control_values=(1,))))
    
    ops.append(qml.Hadamard(wires=qsvt_wires[0]))

    return ops

def qsvt_ls_pyqsp_list(A, b, phi, qsvt_wires):
    ops = []
    ops.append(qml.StatePrep(b, wires=qsvt_wires[2:]))
    ops += real_pyqsp_list(A.T, phi, qsvt_wires)  # invert the singular values of A transpose to get A^-1

    return ops

class qsvt_ls_pyqsp(qml.operation.Operation):
    def __init__(self, A, b, phi, wires, id=None):
        super().__init__(wires=wires, id=id)
        self.A = A
        self.b = b
        self.phi = phi

    def decomposition(self):
        # Define the sequence of operations that make up the custom operation
        return qsvt_ls_pyqsp_list(self.A, self.b, self.phi, self.wires)

def plot_poly_pyqsp(phi, s, kappa):
    x_vals = np.linspace(0, 1, 50)
    target_y_vals = [s * (1 / x) for x in np.linspace(s, 1, 50)]

    qsvt_y_vals = []
    for x in x_vals:
        poly_x = qml.matrix(qml.qsvt, wire_order=[0])(
            x, phi, wires=[0], convention="Wx" 
        )
        qsvt_y_vals.append(np.real(poly_x[0, 0]))
    
    plt.plot(x_vals, np.array(qsvt_y_vals), label="Re(qsvt)")
    plt.plot(np.linspace(s, 1, 50), target_y_vals, label="target")

    plt.vlines(1 / kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
    plt.vlines(0.0, -1.0, 1.0, color="black")
    plt.hlines(0.0, -0.1, 1.0, color="black")

    plt.legend()
    plt.show()


# QSVT with QSPPACK polynomial (cf. https://github.com/qsppack/QSPPACK)

def real_QSPPACK_list(A, phi, qsvt_wires, deg):
    if deg % 2 == 0:
        phi_even = [-x + np.pi / 2 for x in phi]
        phi_even[0] += np.pi / 4
        phi_even[-1] -= 3 * np.pi / 4
    else:
        phi_even = phi
    
    ops = []

    ops.append(qml.Hadamard(wires=qsvt_wires[0]))

    ops.append(qml.ctrl(qml.qsvt(A, phi, qsvt_wires[1:]), control=qsvt_wires[0], control_values=(0,)))
    ops.append(qml.ctrl(qml.qsvt(A.T, phi_even, qsvt_wires[1:]), control=qsvt_wires[0], control_values=(1,)))

    ops.append(qml.Hadamard(wires=qsvt_wires[0]))

    return ops

def qsvt_ls_QSPPACK_list(A, b, phi, qsvt_wires, deg):
    ops = []
    if deg % 4 == 0:
        ops.append(qml.GlobalPhase(-np.pi / 2))
    elif deg % 4 == 2:
        ops.append(qml.GlobalPhase(np.pi / 2))
    elif deg % 4 == 3:
        ops.append(qml.GlobalPhase(-np.pi))
    
    ops.append(qml.StatePrep(b, wires=qsvt_wires[2:]))
    ops += real_QSPPACK_list(A.T, phi, qsvt_wires, deg)  # invert the singular values of A transpose to get A^-1

    return ops

class qsvt_ls_QSPPACK(qml.operation.Operation):
    def __init__(self, A, b, phi, wires, deg, id=None):
        super().__init__(wires=wires, id=id)
        self.A = A
        self.b = b
        self.phi = phi
        self.deg = deg

    def decomposition(self):
        # Define the sequence of operations that make up the custom operation
        return qsvt_ls_QSPPACK_list(self.A, self.b, self.phi, self.wires, self.deg)
    
def qsvt_QSPPACK_list(A, phi, qsvt_wires, deg):
    ops = []
    if deg % 4 == 0:
        ops.append(qml.GlobalPhase(-np.pi / 2))
    elif deg % 4 == 2:
        ops.append(qml.GlobalPhase(np.pi / 2))
    elif deg % 4 == 3:
        ops.append(qml.GlobalPhase(-np.pi))
    
    ops += real_QSPPACK_list(A.T, phi, qsvt_wires, deg)  # invert the singular values of A transpose to get A^-1

    return ops

class qsvt_QSPPACK(qml.operation.Operation):
    def __init__(self, A, phi, wires, deg, id=None):
        super().__init__(wires=wires, id=id)
        self.A = A
        self.phi = phi
        self.deg = deg

    def decomposition(self):
        # Define the sequence of operations that make up the custom operation
        return qsvt_QSPPACK_list(self.A, self.phi, self.wires, self.deg)

def QSPPACK_qsvt(x, phi, wires, deg):
    if deg % 4 == 0:
        qml.GlobalPhase(-np.pi / 2)
    elif deg % 4 == 2:
        qml.GlobalPhase(np.pi / 2)
    elif deg % 4 == 3:
        qml.GlobalPhase(-np.pi)

    qml.qsvt(x, phi, wires)

def plot_poly_QSPPACK(phi, s, kappa, degree):
    x_vals = np.linspace(s, 1, 50)
    inv_x = [s * (1 / x) for x in x_vals]

    samples_x = np.linspace(0, 1, 100)

    qsvt_y_vals = []
    for x in samples_x:
        poly_x = qml.matrix(QSPPACK_qsvt, wire_order=[0])(
            x, phi, wires=[0], deg=degree)
        qsvt_y_vals.append(np.real(poly_x[0, 0]))

    plt.plot(samples_x, qsvt_y_vals, label="Re(qsvt)")
    plt.plot(x_vals, inv_x, label="target")

    plt.vlines(1 / kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
    plt.vlines(0.0, -1.0, 1.0, color="black")
    plt.hlines(0.0, -0.1, 1.0, color="black")

    plt.legend()
    plt.show()

def get_precalculated_angles(degree):
    file_path = '../Phi.txt'

    if degree % 2 == 0 or degree < 1 or degree > 999:
        raise ValueError("Degree is not a precalculated degree. Insert odd number between 1-999.")

    phi_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if f'Degree {degree}:' in line:
                phi_values = [float(value) for value in line.split(':')[1].strip('[] \n').split(',')]
                break

    return phi_values