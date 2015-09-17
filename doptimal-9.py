from lib import *


if __name__ == '__main__':
    aa = [1, 1, 1, 1, 1, 1, 1, -1, -1]
    bb = [1, 1, -1, 1, -1, 1, 1, 1, -1]
    a = sum(aa)
    b = sum(bb)

    print "a = {0}, b = {1}".format(sum(aa), sum(bb))
    print ""
    print "Checking diophantine equation (expect 34)"
    print "a^2 + b^2 = {0}".format(sum(aa) ** 2 + sum(bb) ** 2)
    print ""
    print "Checking PSDs (expect sum to be 16)"

    for s in range(len(aa)):
        psd_a = psd(aa, s)
        psd_b = psd(bb, s)
        print "S {3:2}    PSD(A) {0:.5f}  PSD(B) {1:.5f}  PSD(A)+PSD(B) {2:.5f}".format(
            psd_a, psd_b, psd_a+psd_b, s
        )

    print ""
    print "Checking PAF (expect sum to be 2)"

    for s in range(len(aa)):
        paf_a = paf(aa, s)
        paf_b = paf(bb, s)
        print "S {3:2}    PAF(A) {0:.5f}  PAF(B) {1:8.5f}  PAF(A)+PAF(B) {2:.5f}".format(
            paf_a, paf_b, paf_a+paf_b, s
        )
