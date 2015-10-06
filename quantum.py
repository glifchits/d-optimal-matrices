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


def gate(qubits, unitary, qidx):
    gate_seq = (unitary if idx == qidx else I for idx in range(qubits))
    return kron(*gate_seq)


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

    def apply_gate(self, gate):
        new_state = gate * self.state
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
        return (self.state == other.state).all()


import unittest


class TestQ(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()

