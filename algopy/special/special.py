import math

import numpy
import scipy
import scipy.special

def dpm_hyp1f1(a, b, x):
    """
    generic implementation of

    y = dpm_hyp1f1(a, b, x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """
    try:
        import mpmath
    except ImportError:
        raise Exception('you need to install mpmath to use dpm_ functions')

    #FIXME: move this function?
    def _float_dpm_hyp1f1(a_in, b_in, x_in):
        value = mpmath.hyp1f1(a_in, b_in, x_in)
        try:
            return float(value)
        except:
            return numpy.nan
    _dpm_hyp1f1 = numpy.vectorize(_float_dpm_hyp1f1)

    if hasattr(x.__class__, 'dpm_hyp1f1'):
        return x.__class__.dpm_hyp1f1(a, b, x)
    else:
        return _dpm_hyp1f1(a, b, x)

#dpm_hyp1f1.__doc__ += scipy.special.hyp1f1.__doc__


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


def dpm_hyp2f0(a1, a2, x):
    """
    generic implementation of

    y = dpm_hyp2f0(a1, a2, x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """
    try:
        import mpmath
    except ImportError:
        raise Exception('you need to install mpmath to use dpm_ functions')

    #FIXME: move this function?
    def _float_dpm_hyp2f0(a1_in, a2_in, x_in):
        value = mpmath.hyp2f0(a1_in, a2_in, x_in)
        try:
            return float(value)
        except:
            return numpy.nan
    _dpm_hyp2f0 = numpy.vectorize(_float_dpm_hyp2f0)

    if hasattr(x.__class__, 'dpm_hyp2f0'):
        return x.__class__.dpm_hyp2f0(a1, a2, x)
    else:
        return _dpm_hyp2f0(a1, a2, x)

#dpm_hyp2f0.__doc__ += scipy.special.hyp2f0.__doc__


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

#FIXME: the functions have different calling conventions, so different docs
#hyp2f0.__doc__ += scipy.special.hyp2f0.__doc__


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

    #FIXME: this works around two scipy.special.hyp0f1 failures
    def _hacked_hyp0f1(b, x):
        with numpy.errstate(invalid='ignore'):
            return scipy.special.hyp0f1(b, x + 0j)

    if hasattr(x.__class__, 'hyp0f1'):
        return x.__class__.hyp0f1(b, x)
    else:
        return _hacked_hyp0f1(b, x)

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


def dawsn(x):
    """
    generic implementation of

    y = dawsn(x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'dawsn'):
        return x.__class__.dawsn(x)
    else:
        return scipy.special.dawsn(x)

dawsn.__doc__ += scipy.special.dawsn.__doc__


def logit(x):
    """
    generic implementation of

    y = logit(x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'logit'):
        return x.__class__.logit(x)
    else:
        return scipy.special.logit(x)

logit.__doc__ += scipy.special.logit.__doc__


def expit(x):
    """
    generic implementation of

    y = expit(x)

    x:      either a

            * float
            * numpy.ndarray
            * algopy.UTPM
            * algopy.Function

            instance.


    """

    if hasattr(x.__class__, 'expit'):
        return x.__class__.expit(x)
    else:
        return scipy.special.expit(x)

expit.__doc__ += scipy.special.expit.__doc__


