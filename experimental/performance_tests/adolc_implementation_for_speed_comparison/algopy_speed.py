#!/usr/bin/env python
import sys
import numpy
import numpy.linalg
import time

sys.path = ['../..'] + sys.path
from vector_forward_mode import *

def speelpenning(x):
	return numpy.prod(x)


N = 100
P = N*(N+1)/2
D = 3

x = numpy.zeros(N)
S = numpy.zeros((N,P))

ax = double_to_adouble(x,S,D)

start_time = time.time()
speelpenning(ax)
end_time =time.time()

print('needed time was %0.6f seconds!'%(end_time-start_time))