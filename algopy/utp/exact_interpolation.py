"""
This file implements EXACT INTERPOLATION

The theory is explained on page 315 of the book "Evaluating Derivatives" by Andreas Griewank,
Chapter 13, Subsection: Multivariate Tensors via Univariate Tensors.

Or more in depth in the paper "Evaluating higher derivative tensors by forward propagation of univariate Taylor series"
by  Andreas Griewank, Jean Utke, Andrea Walther

"""
from __future__ import division
import numpy

def generate_multi_indices(N,D):
    """
    generates 2D array of all possible multi-indices with |i| = D
    e.g.
    N=3, D=2
    array([[2, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2]])
    i.e. each row is one multi-index.
    """
    T = []
    def rec(r,n,N,D):
        j = r.copy()
        if n == N-1:
            j[N-1] = D - numpy.sum(j[:])
            T.append(j.copy())
            return
        for a in range( D - numpy.sum( j [:] ), -1,-1 ):
            j[n]=a
            rec(j,n+1,N,D)
    r = numpy.zeros(N,dtype=int)
    rec(r,0,N,D)
    return numpy.array(T)


def multi_index_binomial(z,k):
    """n and k are multi-indices, i.e.
    n = [n1,n2,...]
    k = [k1,k2,...]
    and computes
    n1!/[(n1-k1)! k1!] * n2!/[(n2-k2)! k2!] * ....
    """
    def binomial(z,k):
        """ computes z!/[(z-k)! k!] """
        u = int(numpy.prod([z-i for i in range(k) ]))
        d = numpy.prod([i for i in range(1,k+1)])
        return u/d

    assert numpy.shape(z) == numpy.shape(k)
    N = numpy.shape(z)[0]

    return numpy.prod([ binomial(z[n],k[n]) for n in range(N)])

def multi_index_abs(z):
    return numpy.sum(z)


def convert_multi_indices_to_pos(in_I):
    """
    a multi-index [2,1,0] tells us that we differentiate twice w.r.t x[0] and once w.r.t
    x[1] and never w.r.t x[2]
    This multi-index represents therefore the [0,0,1] element in the derivative tensor.
    """
    I = in_I.copy()
    M,N = numpy.shape(I)
    D = numpy.sum(I[0,:])
    retval = numpy.zeros((M,D),dtype=int)
    for m in range(M):
        i = 0
        for n in range(N):
            while I[m,n]>0:
                retval[m,i]=n
                I[m,n]-=1
                i+=1
    return retval

def gamma(i,j):
    """ Compute gamma(i,j), where gamma(i,j) is define as in Griewanks book in Eqn (13.13)"""
    N = len(i)
    D = numpy.sum(j)
    retval = [0.]
        
    def binomial(z,k):
        """ computes z!/[(z-k)! k!] """
        u = int(numpy.prod([z-i for i in range(k) ]))
        d = int(numpy.prod([i for i in range(1,k+1)]))
        return u//d
    
    def alpha(i,j,k):
        """ computes one element of the sum in the evaluation of gamma,
        i.e. the equation below 13.13 in Griewanks Book"""
        term1 = (1-2*(numpy.sum(abs(i-k))%2))
        term2 = 1
        for n in range(N):
            term2 *= binomial(i[n],k[n])
        term3 = 1
        for n in range(N):
            term3 *= binomial(D*k[n]// numpy.sum(abs(k)), j[n] )
        term4 = (numpy.sum(abs(k))/D)**(numpy.sum(abs(i)))
        return term1*term2*term3*term4
        
    def sum_recursion(in_k, n):
        """ computes gamma(i,j).
            The summation 0<k<i, where k and i multi-indices makes it necessary to do this 
            recursively.
        """
        k = in_k.copy()
        if n==N:
            retval[0] += alpha(i,j,k)
            return
        for a in range(i[n]+1):
            k[n]=a
            sum_recursion(k,n+1)
            
    # putting everyting together here
    k = numpy.zeros(N,dtype=int)
    sum_recursion(k,0)
    return retval[0]
    
def generate_permutations(in_x):
    x = in_x[:]
    if len(x) <=1:
        yield x
    else:
        for perm in generate_permutations(x[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + x[0:1] + perm[i:]
                
                
def generate_Gamma(i):
    """
    generates a big matrix Gamma with elements gamma(i,j)
    
    e.g. 
    i  = [1,3,0,6]
    means that a function f: R^4 -> R^M should be differentiated as
    
    d^10f/(dx1^1 dx2^3 dx4^6)
    
    to do so, the following univariate Taylor polynomials are propagated
    f(x + j t)
    
    INPUTS:
    i           (N,) int-array            multiindex to compute f_i
    
    OUTPUTS:
    J           (Nj,N) int-array            rayvalues
    Gamma       (NJ,)  int-array            interpolation matrix
    """
    
    i = numpy.asarray(i, dtype=int)
    
    if i.ndim != 1:
        raise ValueError('Expected 1D array but provided i.shape = ',i.shape)
    
    N = i.size
    D = numpy.sum(i)
    J = generate_multi_indices(N,D)
    NJ = J.shape[0]
    print N,D,NJ
    
    print 'i=',i
    
    out = numpy.zeros(NJ)
    for nj in range(NJ):
        j = J[nj,:]
        print 'j=',j
        print gamma(i,j)
        out[nj] = gamma(i,j)
            
    return (out,J)
