Comparison and Combination of Forward and Reverse Mode
======================================================

We show here how the forward and the reverse mode of AD are used and show
that they produce the same result. It is also shown how the forward and the
reverse mode can be combined to compute the Hessian of a function

We consider the function :math:`f:\mathbb R^N \times \mathbb R^N\rightarrow \mathbb R` defined by

.. math::
    x,y \mapsto z = x^T y +  (x \circ y - x)^T (x-y)
    
We want to compute the Hessian of that function. The following code shows how
this can be accomplished by a combined forward/reverse computation.

.. literalinclude:: comparison_forward_reverse_mode.py

