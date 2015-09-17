# -*- coding=utf-8 -*-

N = 21

import sys
from lib import psd, paf


def equal(a, b):
    EPSILON = 0.00001
    return abs(a - b) < EPSILON


def seq_to_str(seq):
    c = map(lambda x: '+' if x == 1 else '-', seq)
    return ''.join(c)


def all_possible_sequences(n):
    masks = [1 << j for j in xrange(n)]
    seqs = []
    for i in xrange(2 ** n):
        yield [1 if (masks[j] & i) else -1 for j in xrange(n)]


def check_diophantine_invariant(A, B):
    v = len(A)
    a = sum(A)
    b = sum(B)
    # p.279 in "New Results..."
    #
    # By pre- and post-multiplying equation (1) with J_v, one obtains that
    #   a^2 + b^2 = 4v−2,
    # where a and b are row sums of A and B
    #
    return equal(a ** 2 + b ** 2, 4*v - 2)


def check_paf_invariant(A, B):
    for s in xrange(1, len(A)/2):
        paf_a = paf(A, s)
        paf_b = paf(B, s)
        if not equal(paf_a + paf_b, 2):
            return False
    return True


def check_psd_invariant(A, B):
    v = len(A)
    # p.280 in "New Results..."
    #
    # define a circulant D-optimal matrix implies that the sum
    # PSD_A(s) + PSD_B(s), s ≠ 0, is equal to the constant 2v − 2
    #
    cond = 2*v - 2

    for k in xrange(1, v):
        # non negativity of PSD. we can discard before computing PSD both times
        psd_a = psd(A, k)
        if psd_a > cond:
            return False
        psd_b = psd(B, k)
        if psd_b > cond:
            return False
        if not equal(psd_a + psd_b, cond):
            return False
    return True


matches = []
iterations = 0
max_possible = (2**N)**2

print "max_possible:", max_possible

for aa in all_possible_sequences(N):
    for bb in all_possible_sequences(N):
        iterations += 1
        percent_done = iterations / max_possible
        if iterations % 1000 == 0:
            print ".",
        # check that a^2 + b^2 = 34
        r = check_diophantine_invariant(aa, bb)
        if not r: continue
        # check the PAF invariant
        r = check_paf_invariant(aa, bb)
        if not r: continue
        r = check_psd_invariant(aa, bb)
        if not r: continue
        if max_possible > 100000:
            print "\nfound sequences!"
            print "A:", seq_to_str(aa)
            print "B:", seq_to_str(bb)
            sys.exit()
        # all invariants hold!
        matches.append((aa, bb))


print "Found {0} sequences out of {1} possible.".format(len(matches), (2**N)**2)
if len(matches) == 0:
    sys.exit()

first = matches[0]
print "First match"
print "A:", seq_to_str(first[0])
print "B:", seq_to_str(first[1])
