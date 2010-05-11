#!/usr/bin/env python

from numpy import *
from adolc import *


class SparseMatrix:
    """ Sparse Matrix as Vertex List"""
    def __init__(self,N_rows,N_cols):
        self.N = N_cols
        self.M = N_rows
        self.R = []
    def __repr__(self):
        return str(self.R)
    def __str__(self):
        return self.__repr__()


def mul(Z,X,Y):
    """
    Z = dot(X,Y), where Z,X,Y sparse matrices
    the bitpattern of Z is given, i.e. not all operations in dot(X,Y) are necessary.
    This multiplication function takes advantage of this knowledge.
    """


if __name__ == "__main__":
    N = 2
    A = SparseMatrix(N,N)
    A.R.append((0,0,12.))
    A.R.append((1,0,13.))

    print A