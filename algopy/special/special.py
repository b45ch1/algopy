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


def erfi(x):
    """
    generic implementation of

    y = erfi(x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'erfi'):
        return x.__class__.erfi(x)
    else:
        #FIXME: scipy.special.erfi does not yet exist
        #return scipy.special.erfi(x)
        return 2 * x * scipy.special.hyp1f1(0.5, 1.5, x*x) / (
                math.sqrt(math.pi))

#FIXME: scipy.special.erfi does not yet exist
#erfi.__doc__ += scipy.special.erfi.__doc__

