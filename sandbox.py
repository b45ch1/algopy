import numpy
import numpy.random

from algopy import UTPM, Function, CGraph, sum, zeros, diag, dot, qr



D,P,N,M = 2,3,4,5

X = 2 * numpy.random.rand(D,P,N,M)
Y = 3 * numpy.random.rand(N,M)
AX = UTPM(X)
# AY1 = Y + AX
# AY2 = Y - AX
# AY3 = Y * AX
AY4 = Y / AX
# AY5 = AX + Y
# AY6 = AX - Y
# AY7 = AX * Y
# AY8 = AX / Y