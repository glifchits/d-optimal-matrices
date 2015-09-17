import cmath


def paf(seq, i):
    n = len(seq)
    pairs = [seq[k] * seq[(k+i) % n] for k in range(n)]
    return sum(pairs)


def dft(seq, s):
    n = len(seq)
    omega = cmath.e ** ((2 * cmath.pi * 1j) / n)
    terms = [seq[k] * omega ** (k+s) for k in range(n)]
    return sum(terms)


def psd(seq, s):
    n = len(seq)
    d = dft(seq, s)
    return d.real ** 2 + d.imag ** 2

