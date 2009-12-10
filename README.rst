ALGOPY, a library for Automatic Differentation (AD) in Python
-------------------------------------------------------------

Rationale:
    ALGOPY is a research prototype striving to provide state of the art algorithms.
    It is not (yet) geared towards end users.
    The ultimative goal is to provide high performance algorithms
    that can be used to differentiate dynamic systems  (ODEs, DAEs, PDEs)
    and static systems (linear/nonlinear systems of equations).
    
    ALGOPY focuses on the algebraic differentiation of elementary operations,
    e.g. C = dot(A,B) where A,B,C are matrices, y = sin(x), z = x*y, etc.
    to compute derivatives of functions composed of such elementary functions.
    
    In particular, ALGOPY offers:
        
        Univariate Taylor Propagation:
            
            * Univariate Taylor Propagation on Scalars  (UTPS)
              Implementation in: `./algopy/utp/utps.py`
            * Univariate Taylor Propagation on Matrices (UTPM)
              Implementation in: `./algopy/utp/utpm.py`
            * Cross Taylor Propagation on Scalars (CPTS)
              Implementation in: `./algopy/utp/ctps_c.py`
            * Exact Interpolation of Higher Order Derivative Tensors:
              (Hessians, etc.)
              
        Reverse Mode:
        
            ALGOPY also features functionality for convenient differentiation of a given
            algorithm. For that, the sequence of operation is recorded by tracing the 
            evaluation of the algorithm. Implementation in: `./algopy/tracer.py`

    ALGOPY aims to provide algorithms in a clean and accessible way allowing quick
    understanding of the underlying algorithms. Therefore, it should be easy to
    port to other programming languages, take code snippets.
    If optimized algorithms are wanted, they should be provided in a subclass derived
    from the reference implementation.
    

Dependencies:
    ALGOPY Core:
        * numpy

    ALGOPY Examples:
        * pyadolc
        * scipy

    Run tests:
        * Nose

Alternatives:
    If you are looking for a robust tool for AD in Python you should try:
        
        * `PYADOLC`_ a Python wrapper for ADOL-C (C++)
        * `PYCPPAD`_ a Python wrapper for  CppAD (C++)

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