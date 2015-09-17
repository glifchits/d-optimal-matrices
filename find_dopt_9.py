N = 9

from lib import psd, paf


def equal(a, b):
    EPSILON = 0.00001
    return abs(a - b) < EPSILON


def all_possible_sequences(n):
    masks = [1 << j for j in xrange(n)]
    seqs = []
    for i in xrange(2 ** n):
        yield [1 if (masks[j] & i) else -1 for j in xrange(n)]


def check_diophantine_invariant(A, B):
    a = sum(A)
    b = sum(B)
    return equal(a ** 2 + b ** 2, 34)


def check_paf_invariant(A, B):
    for s in xrange(1, len(A)):
        paf_a = paf(A, s)
        paf_b = paf(B, s)
        if not equal(paf_a + paf_b, 2):
            return False
    return True


def check_psd_invariant(A, B):
    for s in xrange(len(A)):
        psd_a = psd(A, s)
        psd_b = psd(B, s)
        if not equal(psd_a + psd_b, 16):
            return False
    return True


matches = []


for aa in all_possible_sequences(N):
    for bb in all_possible_sequences(N):
        # check that a^2 + b^2 = 34
        r = check_diophantine_invariant(aa, bb)
        if not r: continue
        # check the PAF invariant
        r = check_paf_invariant(aa, bb)
        if not r: continue
        r = check_psd_invariant(aa, bb)
        if not r: continue
        # all invariants hold!
        matches.append((aa, bb))


print "Found {0} sequences out of {1} possible.".format(len(matches), (2**N)**2)


a = [1, 1, 1, 1, 1, 1, 1, -1, -1]
b = [1, 1, -1, 1, -1, 1, 1, 1, -1]
for aa, bb in matches:
    if aa == a and bb == b:
        print "-" * 80
        print "Matching!"
        print "A", aa
        print "B", bb
        print "-" * 80

