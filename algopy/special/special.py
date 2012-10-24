import math

import numpy
import scipy
import scipy.special

def hyp1f1(a, b, x):
    """
    generic implementation of

    y = hyp1f1(a, b, x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'hyp1f1'):
        return x.__class__.hyp1f1(a, b, x)
    else:
        return scipy.special.hyp1f1(a, b, x)

hyp1f1.__doc__ += scipy.special.hyp1f1.__doc__


def hyp2f0(a1, a2, x):
    """
    generic implementation of

    y = hyp2f0(a1, a2, x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'hyp2f0'):
        return x.__class__.hyp2f0(a1, a2, x)
    else:
        # FIXME: use convergence_type 1 vs. 2 ?  Scipy docs are not helpful.
        convergence_type = 2
        value, error_info = scipy.special.hyp2f0(a1, a2, x, convergence_type)
        return value

hyp2f0.__doc__ += scipy.special.hyp2f0.__doc__


def hyp0f1(b, x):
    """
    generic implementation of

    y = hyp0f1(b, x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'hyp0f1'):
        return x.__class__.hyp0f1(b, x)
    else:
        return scipy.special.hyp0f1(b, x)

hyp0f1.__doc__ += scipy.special.hyp0f1.__doc__


def erf(x):
    """
    generic implementation of

    y = erf(x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'erf'):
        return x.__class__.erf(x)
    else:
        return scipy.special.erf(x)

erf.__doc__ += scipy.special.erf.__doc__



