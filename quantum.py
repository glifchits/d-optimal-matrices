import numpy as np
import random
from math import log
from cmath import sqrt
from functools import reduce


def kron(*matrices):
    return reduce(np.kron, matrices[1:], matrices[0])


def row_swap(matrix_to_swap, i, j):
    matrix = np.matrix(np.copy(matrix_to_swap))
    tmp = np.copy(matrix[i])
    matrix[i] = np.copy(matrix[j])
    matrix[j] = tmp
    return matrix


zero = np.matrix('1; 0')
one = np.matrix('0; 1')

I = np.eye(2, dtype=int)
X = np.matrix('0 1; 1 0')
Z = np.matrix('1 0; 0 -1')
H = 1/sqrt(2) * (X + Z)
entangled_pair = 1/sqrt(2) * (kron(zero, zero) + kron(one, one))

cnot_10 = np.matrix('1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0')
cnot_01 = np.matrix('1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0')


def get_entangled_pair():
    return Q(entangled_pair)


class Q(object):

    def __init__(self, state):
        state_as_row = state.flatten().tolist()[0]
        sq_mag = sum(abs(a)**2 for a in state_as_row)
        assert abs(1-sq_mag) < 0.00001, "Squared magnitudes must sum to 1"
        self.state = state

    def __add__(self, other):
        new_state = self.state + other.state
        as_row = new_state.flatten().tolist()[0]
        norm_factor = sum(p**2 for p in as_row)
        norm_new_state = 1/norm_factor * new_state
        return Q(norm_new_state)

    def apply_gate(self, gate, qubit_idx):
        # construct a unitary transformation matrix of order `num_qubits`
        # `qubits` is a reversed range since qubits index from right to left
        qubits = reversed(range(self.num_qubits))
        gate_seq = (gate if i == qubit_idx else I for i in qubits)
        unitary_transform = kron(*gate_seq)
        # multiply the state with this unitary transform
        new_state = unitary_transform * self.state
        # renormalize the state
        as_row = new_state.flatten().tolist()[0]
        norm_factor = sum(p**2 for p in as_row)
        norm_new_state = 1/norm_factor * new_state
        return Q(norm_new_state)

    def apply_cnot(self, control, target):
        control_mask = 2 ** control
        target_mask = 2 ** target
        new_state = self.state
        swapped = set([])
        for qubit in range(2 ** self.num_qubits):
            # only consider qubits (states) where the control bit is > 0
            if qubit & control_mask == 0: continue
            # compute the target bit index
            target_idx = qubit ^ target_mask
            # this method calls for (i, j) and (j, i) to be swapped
            # to prevent double-swapping, check if (i, j) have not already
            # been swapped
            if (target_idx, qubit) in swapped: continue
            # perform the row swap
            new_state = row_swap(new_state, qubit, target_idx)
            # add both (i, j) and (j, i) to the swapped set
            swapped.add((target_idx, qubit))
            swapped.add((qubit, target_idx))
        return Q(new_state)

    @property
    def num_qubits(self):
        rows, cols = self.state.shape
        qubits = log(rows, 2)
        assert qubits % 1.0 == 0, "Got irregular number of qubits"
        return int(qubits)

    def measure(self):
        amplitudes = self.state.flatten().tolist()[0]
        probabilities = ((a**2).real for a in amplitudes)
        cumul = 0
        rand = random.random()
        for idx, pdensity in enumerate(probabilities):
            cumul += pdensity
            if rand <= cumul:
                return idx
        raise AssertionError("probabilities did not sum to 1")

    def __repr__(self):
        return str(self.state)

    def __eq__(self, other):
        # the matrix equality comparator doesn't allow rounding error.
        # see docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
        return np.allclose(self.state, other.state)


import unittest


class TestQ(unittest.TestCase):

    def test_num_qubits_1(self):
        state = Q(zero)
        self.assertEqual(state.num_qubits, 1)

    def test_num_qubits_2(self):
        state = Q(kron(zero, zero))
        self.assertEqual(state.num_qubits, 2)

    def test_num_qubits_3(self):
        QUBITS = 10
        state = Q(kron(*(zero for x in range(QUBITS))))
        self.assertEqual(state.num_qubits, QUBITS)

    def test_cnot_1(self):
        state = Q(kron(one, zero))
        state = state.apply_cnot(1, 0)
        exp_state = Q(kron(one, one))
        self.assertEqual(state, exp_state)

    def test_cnot_2(self):
        state = Q(kron(one, zero))
        state = state.apply_cnot(0, 1)
        exp_state = Q(kron(one, zero))
        self.assertEqual(state, exp_state)

    def test_cnot_3(self):
        state = Q(kron(one, one))
        state = state.apply_cnot(0, 1)
        exp_state = Q(kron(zero, one))
        self.assertEqual(state, exp_state)

    def test_cnot_4(self):
        state = Q(kron(one, one, one))
        state = state.apply_cnot(2, 0)
        exp_state = Q(kron(one, one, zero))
        self.assertEqual(state, exp_state)

    def test_cnot_5(self):
        state = Q(kron(one, one, zero))
        state = state.apply_cnot(2, 0)
        exp_state = Q(kron(one, one, one))
        self.assertEqual(state, exp_state)

    def test_cnot_6(self):
        state = Q(kron(one, one, one))
        state = state.apply_cnot(0, 2)
        exp_state = Q(kron(zero, one, one))
        self.assertEqual(state, exp_state)

    def test_cnot_7(self):
        state = Q(kron(zero, one, one, zero, one))
        state = state.apply_cnot(2, 4)
        exp_state = Q(kron(one, one, one, zero, one))
        self.assertEqual(state, exp_state)

    def test_cnot_8(self):
        state = Q(kron(one, zero, one, one, one, zero, one))
        state = state.apply_cnot(1, 6)
        exp_state = Q(kron(one, zero, one, one, one, zero, one))
        self.assertEqual(state, exp_state)

    def test_cnot_9(self):
        state = Q(kron(one, zero, one, one, one, zero, one))
        state = state.apply_cnot(2, 6)
        exp_state = Q(kron(zero, zero, one, one, one, zero, one))
        self.assertEqual(state, exp_state)

    def test_apply_gate_1(self):
        state = Q(zero)
        state = state.apply_gate(X, 0)
        exp_state = Q(one)
        self.assertEqual(state, exp_state)

    def test_apply_gate_2(self):
        state = Q(kron(zero, zero))
        state = state.apply_gate(X, 0)
        exp_state = Q(kron(zero, one))
        self.assertEqual(state, exp_state)

    def test_apply_gate_3(self):
        state = Q(kron(one, zero))
        state = state.apply_gate(X, 1)
        exp_state = Q(kron(zero, zero))
        self.assertEqual(state, exp_state)

    def test_apply_gate_entangled_pair(self):
        state = Q(kron(zero, zero))
        state = state.apply_gate(H, 1)
        state = state.apply_cnot(1, 0)
        exp_state = Q(entangled_pair)
        self.assertEqual(state, exp_state)


if __name__ == '__main__':
    unittest.main()

