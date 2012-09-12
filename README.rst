AlgoPy, a library for Automatic Differentation (AD) in Python
-------------------------------------------------------------

Description:
    AlgoPy allows you to differentiate functions implemented as computer programs
    by using Algorithmic Differentiation (AD) techniques in the forward and
    reverse mode.

    The forward mode propagates univariate Taylor polynomials of arbitrary order.
    Hence it is also possible to use AlgoPy to evaluate higher-order derivative tensors.

    Speciality of AlgoPy is the possibility to differentiate functions that contain
    matrix functions as +,-,*,/, dot, solve, qr, eigh, cholesky.


Rationale:
    Many programs for scientific computing make use of numerical linear algebra.
    The defacto standard for array manipulations in Python is NumPy.
    AlgoPy allows you to write code that can either be evaluated by NumPy, or with
    AlgoPy with little or no modifications to your code.

    Note that this does not mean that any code you wrote can be differentiated with AlgoPy,
    but rather that you can write code that can be evaluated with or without AlgoPy.


Documentation:
    Available at http://packages.python.org/algopy/

    For more documentation have a look at:
        1) the talks in the ./documentation folder
        2) the examples in the ./documentation/examples folder
        3) sphinx documenation ./documentation/sphinx and run `make`


Example:
    Compute directional derivatives of the function f(J)::

        import numpy
        from algopy import UTPM, qr, solve, dot, eigh

        def f(x):
            N,M = x.shape
            Q,R = qr(x)
            Id = numpy.eye(M)
            Rinv = solve(R,Id)
            C = dot(Rinv,Rinv.T)
            l,U = eigh(C)
            return l[0]

        x = UTPM.init_jacobian(numpy.random.random((50,10)))
        y = f(x)
        J = UTPM.extract_jacobian(y)

        print 'Jacobian dy/dx =', J



Features:

    Univariate Taylor Propagation:

        * Univariate Taylor Propagation on Matrices (UTPM)
          Implementation in: `algopy.utpm`
        * Exact Interpolation of Higher Order Derivative Tensors:
          (Hessians, etc.)

    Reverse Mode:

        ALGOPY also features functionality for convenient differentiation of a given
        algorithm. For that, the sequence of operation is recorded by tracing the
        evaluation of the algorithm. Implementation in: `./algopy/tracer.py`

Testing:

    Uses numpy testing facilities. Simply run::

        $ python -c "import algopy; algopy.test()"


Dependencies:

    ALGOPY Core:
        * numpy
        * scipy

    ALGOPY Examples:
        * pyadolc

    Run tests:
        * Nose

    Documentation:
        * sphinx
        * matplotlib, mayavi2, yapgvb

Alternatives:

    If you are looking for a robust tool for AD in Python you should try:

        * `PYADOLC`_ a Python wrapper for ADOL-C (C++)
        * `PYCPPAD`_ a Python wrapper for  CppAD (C++)

    However, their support for differentiation of Numerical Linear Algebra (NLA)
    functions is only very limited.

    .. _PYADOLC: http://www.github.com/b45ch1/pyadolc
    .. _PYCPPAD: http://www.github.com/b45ch1/pycppad

Email:
    sebastian.walter@gmail.com



-------------------------------------------------------------------------------

Licence:
    BSD style using http://www.opensource.org/licenses/bsd-license.php template
    as it was on 2009-01-24 with the following substutions:

    * <YEAR> = 2008-2009
    * <OWNER> = Sebastian F. Walter, sebastian.walter@gmail.com
    * <ORGANIZATION> = contributors' organizations
    * In addition, "Neither the name of the contributors' organizations" was changed to "Neither the names of the contributors' organizations"


Copyright (c) 2008-2009, Seastian F. Walter
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the names of the contributors' organizations nor the names of
      its contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.