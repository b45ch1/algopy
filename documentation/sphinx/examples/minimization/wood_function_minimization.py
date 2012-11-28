"""
Minimize the Wood function.

Problem 3.1 of
"A truncated Newton method with nonmonotone line search
for unconstrained optimization"
Except that I think there is a typo in that paper,
and the minimum is actually at (1,1,1,1) rather than at (0,0,0,0).
"""

import numpy

import minhelper


def wood(X):
    """
    This R^4 -> R^1 function should be compatible with algopy.
    """
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]
    x4 = X[3]
    return sum((
        100*(x1*x1 - x2)**2,
        (x1-1)**2,
        (x3-1)**2,
        90*(x3*x3 - x4)**2,
        10.1*((x2-1)**2 + (x4-1)**2),
        19.8*(x2-1)*(x4-1),
        ))


def main():
    target = [1, 1, 1, 1]
    easy_init = [1.1, 1.2, 1.3, 1.4]
    hard_init = [-3, -1, -3, -1]
    minhelper.show_minimization_results(
            wood, target, easy_init, hard_init)


if __name__ == '__main__':
    main()

