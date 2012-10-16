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

# append the scipy.special docstring
hyp1f1.__doc__ += scipy.special.hyp1f1.__doc__
