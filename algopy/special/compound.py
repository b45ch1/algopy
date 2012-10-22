"""
This file contains the implementation of functions like

algopy.special.erfi

that are not represented as a single node in the
computational graph, but are
treated as a **compound** function.

Note
----

These functions should be replaced by a dedicated implementation in

* algopy.Function
* algopy.UTPM

so they are represented by a single node in the CGraph.

"""

import math
import scipy.special


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

