"""
Implementation of the Cross Derivative Taylor Polynomials
The algebraic class is the factor ring

R[t1,...,tK]/<t1^2,...,tK^2>

where R the real numbers.
"""

import numpy
from algopy.base_type import GradedRing

def i2m(i):
    """
    computes the location in linear memory from the multiindex
    i = (i_1,...i_N)

    size(i) = 2**N
    therefore, N = log2(size(i))

    m(i) = sum([2**n i[n] for n in range(N)])
    """
    N = numpy.size(i)
    return int(numpy.sum([2**n * i[n] for n in range(N)]))

def m2i(m, bits = None):
    """
    computes the multi index from the memory location

    if bits = numpy bool array, then a multi index is generated,
    where the generated i[numpy.where(bits>0)] would be the multi index
    generated without the bits.

    This functionality is needed for the \sum_{k<=i}, where k and i are multiindices

    """

    assert numpy.isscalar(m)
    if bits == None:
        retval = []
        while m != 0:
            retval.append(bool(m%2))
            m/=2
        return numpy.asarray(retval, dtype=bool)
    else:
        N = numpy.size(bits)
        M = numpy.sum(bits)
        retval = numpy.zeros(N,dtype=bool)
        tmp    = numpy.zeros(M,dtype=bool)
        b = numpy.nonzero(bits)
        n = 0
        while m != 0:
            tmp[n] = bool(m%2)
            m/=2
            n +=1
        retval[b] = tmp[:]
        return retval

def inconv(z, x, y):
    N = numpy.size(x)
    if N == 1:
        z[0] += x[0] * y[0]
    else:
        N = N/2
        inconv(z[:N], x[:N], y[:N])
        inconv(z[N:], x[:N], y[N:])
        inconv(z[N:], x[N:], y[:N])

def inconv2(z, x, y):
    Nx = int(numpy.log2(numpy.size(x)))
    i = numpy.ones(Nx,dtype=bool)

    for mi in range(i2m(i)+1):
        j = m2i(mi)
        for mk in range(2**sum(j)):
            k = m2i(mk,j)
            z[i2m(j)] += x[i2m(k)]*y[i2m(j-k)]

#def inconv3(x, y, z):
    #Nx = int(numpy.log2(numpy.size(x)))
    #z[0] = x[0] * y[0]
    #for nx in range(Nx):
        #for m in range(2**(nx)-1):
            #z[2**nx + m] = numpy.sum(x[:m+1]*y[::-1][:m+1])

class CTPS(GradedRing):
    def __init__(self, data):
        """
        CTPS = Cross Derivative Taylor Polynomial
        Implements the factor ring  R[t1,...,tK]/<t1^2,...,tK^2>
        """
        self.data = numpy.array(data)
        
    @classmethod
    def zeros_like(cls, data):
        return numpy.zeros_like(data)

    @classmethod
    def mul(cls, retval_data, lhs_data, rhs_data):
        inconv(retval_data, lhs_data, rhs_data)


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.data) 
