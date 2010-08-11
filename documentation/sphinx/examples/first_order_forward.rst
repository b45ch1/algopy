First Order Derivatives in the Forward Mode of Algorithmic Differentiation
--------------------------------------------------------------------------
In this example we want to show how one can extract derivatives
from the computed univariate Taylor polynomials (UTP). For simplicity we only show
first-order derivatives but ALGOPY also supports the computation of higher-order
derivatives by an interpolation-evaluation approach.

The basic observation is that by use of the chain rule one obtains functions
:math:`F: \mathbb R^N \rightarrow \mathbb R^M`

.. math::
    \left. \frac{d}{d t} F(x_0 + x_1 t) \right|_{t=0} = \left. \frac{d}{d x} f(x) \right|_{x = x_0} \cdot x_1\;.

i.e. a Jacobian-vector product.

Again, we look a simple contrived example and we want to compute the first column
of the Jacobian, i.e., :math:`x_1 = (1,0,0)`.

.. literalinclude:: first_order_forward.py
    :lines: 1-

As output one gets::
    
    y0 =  [  3.  15.   5.]
    y  =  [[[  3.  15.   5.]]
    
     [[  3.   0.   5.]]]
    y.shape = (3,)
    y.data.shape = (2, 1, 3)
    dF/dx(x0) * x1 = [ 3.  0.  5.]


and the question is how to interpret this result. First off, y0 is just the usual
function evaluation using numpy but y represent a univariate Taylor polynomial (UTP).
One can see that each coefficient of the polynomial has the shape (3,). We extract
the directional derivative as the first coefficient of the UTP.

One can see that this is indeed the numerical value of first column of the Jacobian J(1,3,5)::
    
    def J(x):
        ret = numpy.zeros((3,3),dtype=float)
        ret[0,:] = [x[1], x[0],  0  ]
        ret[1,:] = [0,  , x[2], x[1]]
        ret[2,:] = [x[2],   0 , x[0]] 
