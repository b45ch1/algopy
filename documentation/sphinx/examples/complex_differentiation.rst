Differentiation of programs containing complex arithmetic
---------------------------------------------------------

Complex problems are often formulated in simple equations in the complex numbers.
Therefore, quite often codes containing complex arithmetic have to be differentiated.
Algopy has a partial support for complex arithmetic. In this section, we use the simple function


.. math::
    f: \mathbb R^N \to {\mathbb C} \\
        x \mapsto y = f(x) = \sum_{i=1}^N \sum_{j=1}^N A_{ij} x_j

where :math:`A \in {\mathbb C}^{N \times N}`.

In that form, it is not clear what the gradient :math:`\nabla_x f(x)` actually means.
There are two possibilites: either change the signature to :math:`f: {\mathbb C}^N \to {\mathbb C}` and compute the complex derivative,
or reformulate the function 

.. math::
    f: \mathbb R^N \to \mathbb R \\
        x \mapsto y = f(x) = {\mathrm Re} ( \sum_{i=1}^N \sum_{j=1}^N A_{ij} x_j)

to map from reals to reals.


variant 1: :math:`\mathbb C^N \to \mathbb C`::

    import algopy
    import numpy as np

    def f(x, A, module):
          y = module.dot(A, x)
          return module.sum(y)

    size = 4
    Ar = np.random.random((size, size))
    Ai = np.random.random((size, size))
    Ac = Ar +1j*Ai
    A  = Ac
    x  = np.random.random((size,)) + 0j

    cg = algopy.CGraph()
    xf = algopy.Function(x)
    sf = f(xf, A, algopy)
    cg.trace_off()

    print 'sf.x=', sf.x
    assert sf.x == f(x , A, np)

    cg.independentFunctionList = [xf]
    cg.dependentFunctionList = [sf]
    gf = cg.gradient(x)
    ganalytic = np.sum(A, axis=0)

    print 'sf.x=', sf.x
    print 'gf=\n',gf
    print 'np.sum(A, axis=0)=\n',ganalytic
    assert np.allclose(gf, ganalytic)



variant 2: :math:`\mathbb R^N \to \mathbb R`::

    import algopy
    import numpy as np

    def f(x, A, module):
          y = module.dot(A, x)
          return module.real(module.sum(y))

    size = 4
    Ar = np.random.random((size, size))
    Ai = np.random.random((size, size))
    Ac = Ar +1j*Ai
    A  = Ac
    x  = np.random.random((size,))

    cg = algopy.CGraph()
    xf = algopy.Function(x)
    sf = f(xf, A, algopy)
    cg.trace_off()

    print 'sf.x=', sf.x
    assert sf.x == f(x , A, np)

    cg.independentFunctionList = [xf]
    cg.dependentFunctionList = [sf]
    gf = cg.gradient(x)
    ganalytic = np.real(np.sum(A, axis=0))

    print 'gf=\n',gf
    print 'np.sum(A, axis=0)=\n',ganalytic
    assert np.allclose(gf, ganalytic)

