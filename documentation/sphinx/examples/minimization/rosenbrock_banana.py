"""
Minimize the Rosenbrock banana function.

http://en.wikipedia.org/wiki/Rosenbrock_function
"""

import numpy

import minhelper

def rosenbrock(X):
    """
    This R^2 -> R^1 function should be compatible with algopy.
    http://en.wikipedia.org/wiki/Rosenbrock_function
    A generalized implementation is available
    as the scipy.optimize.rosen function
    """
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.

def main():
    target = [1, 1]
    easy_init = [2, 2]
    hard_init = [-1.2, 1]
    minhelper.show_minimization_results(
            rosenbrock, target, easy_init, hard_init)

if __name__ == '__main__':
    main()

