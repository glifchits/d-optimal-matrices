import cmath
from accounting import Accounting


def paf(seq, i):
    Accounting.start_task('paf')
    n = len(seq)
    pairs = [seq[k] * seq[(k+i) % n] for k in range(n)]
    res = sum(pairs)
    Accounting.finish_task('paf')
    return res


def dft(seq, s):
    Accounting.start_task('dft')
    n = len(seq)
    omega = cmath.exp((2 * cmath.pi * 1j) / n)
    terms = [seq[k] * omega ** (k+s) for k in range(n)]
    res = sum(terms)
    Accounting.finish_task('dft')
    return res


def psd(seq, s):
    Accounting.start_task('psd')
    d = dft(seq, s)
    magnitude = d.real ** 2 + d.imag ** 2
    Accounting.finish_task('psd')
    return magnitude

