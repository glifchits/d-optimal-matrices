import cmath
from accounting import Accounting


def paf(seq, i):
    Accounting.start_task('paf')
    n = len(seq)
    res = sum(seq[k] * seq[(k+i) % n] for k in range(n))
    Accounting.finish_task('paf')
    return res


def dft(seq, s):
    Accounting.start_task('dft')
    n = len(seq)
    omega = cmath.exp((2 * cmath.pi * 1j) / n)
    res = sum(seq[k] * omega ** (k+s) for k in range(n))
    Accounting.finish_task('dft')
    return res


def psd(seq, s):
    Accounting.start_task('psd')
    d = dft(seq, s)
    magnitude = d.real ** 2 + d.imag ** 2
    Accounting.finish_task('psd')
    return magnitude

