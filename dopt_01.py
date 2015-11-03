from lib import paf


def int_to_seq(i, N):
    """
    :param i: [int] binary encoding of the sequence
    :param N: [int] desired length of the sequence
    :returns: [list] sequence of 0s and 1s
    """
    bitstr = "{:0{width}b}".format(i, width=N)
    return [int(c) for c in bitstr]


def all_01_seqs(N):
    """
    :param N: desired length of the sequences
    :returns: [generator] all sequences of 0 and 1 length N
    """
    return (int_to_seq(i, N) for i in range(2 ** N))


def all_01_seqpairs(N):
    """
    :param N: desired length of the sequences
    :returns: [generator] all 0/1 sequence pairs of length N
    """
    return ((a,b) for a in all_01_seqs(N) for b in all_01_seqs(N))


def diophantine_invariant(A, B):
    assert len(A) == len(B), "len mismatch"
    N = len(A)
    t1 = (2 * sum(A)) - N
    t2 = (2 * sum(B)) - N
    return (t1 ** 2) + (t2 ** 2) == (4*N - 2)


def paf_invariant(A, B):
    assert len(A) == len(B), "len mismatch"
    N = len(A)
    for i in range(1, 1 + (N-1)//2):
        comp = 2 * (paf(A, i) + paf(B, i) - sum(A) - sum(B)) + N
        if comp != 1: return False
    return True


if __name__ == "__main__":
    import sys
    try:
        N = int(sys.argv[1])
    except:
        sys.stderr.write("You must supply one argument (N)\n")
        sys.exit(1)

    dopt_pairs = []
    iteration = 0
    possible = (2 ** N) ** 2
    log_interval = possible // 23

    for a, b in all_01_seqpairs(N):
        iteration += 1
        if paf_invariant(a, b):
            dopt_pairs.append((a, b))

        if iteration % log_interval == 0:
            print("Checked {:.3f}%".format(100*iteration/possible))

    print("Found {} D-optimal 0/1 pairs length {} out of {} possible.".format(
        len(dopt_pairs), N, possible
    ))
