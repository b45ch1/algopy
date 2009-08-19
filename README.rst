ALGOPY, a library for Automatic Differentation (AD) in Python
-------------------------------------------------------------

Rationale:
    ALGOPY is a research prototype striving to provide state
    of the art algorithms. It is not (yet) geared towards end users.
    The ultimative goal is to provide high performance algorithms
    that can be used to differentiate dynamic systems
    (ODEs, DAEs, PDEs)
    and static systems (linear/nonlinear systems of equations).


Dependencies:
    ALGOPY Core:
        numpy

    ALGOPY Examples:
        * pyadolc
        * scipy

Alternatives:
    If you are looking for a robust tool for AD in Python you should try:
        
        * `PYADOLC`_ a Python wrapper for ADOL-C (C++)
        * `PYCPPAD`_ a Python wrapper for  CppAD (C++)

    .. _PYADOLC: http://www.github.com/b45ch1/pyadolc
    .. _PYCPPAD: http://www.github.com/b45ch1/pycppad





Author and Copyright:
    Sebastian F. Walter

    Copyright (c) 2008-2009.
    All rights reserved.
    
    
Email:
    sebastian.walter@gmail.com





Licence (BSD):
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the HU Berlin nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

Disclaimer:
    THIS SOFTWARE IS PROVIDED BY Sebastian F. Walter  ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Sebastian F. Walter BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.