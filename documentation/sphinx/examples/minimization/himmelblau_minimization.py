"""
Minimize the Himmelblau function.

http://en.wikipedia.org/wiki/Himmelblau%27s_function
"""

import numpy

import minhelper


def himmelblau(X):
    """
    This R^2 -> R^1 function should be compatible with algopy.
    http://en.wikipedia.org/wiki/Himmelblau%27s_function
    This function has four local minima where the value of the function is 0.
    """
    x = X[0]
    y = X[1]
    a = x*x + y - 11
    b = x + y*y - 7
    return a*a + b*b


def main():
    target = [3, 2]
    easy_init = [3.1, 2.1]
    hard_init = [-0.27, -0.9]
    minhelper.show_minimization_results(
            himmelblau, target, easy_init, hard_init)


if __name__ == '__main__':
    main()

