#!/usr/bin/env python
"""ALGOPY: Taylor Arithmetic Computation and Algorithmic Differentiation

ALGOPY is a tool for Algorithmic Differentiation (AD) and Taylor polynomial approximations.
ALGOPY makes it possible to perform computations on scalar and polynomial matrices.
It is designed to be as compatible to numpy as possible. I.e. views, broadcasting and most
functions of numpy can be performed on polynomial matrices. Exampels are dot,trace,qr,solve,
inv,eigh.
The reverse mode of AD is also supported by a simple code evaluation tracer.
"""

DOCLINES = __doc__.split("\n")

import os
import shutil
import sys
import re
import subprocess

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Linux
"""

NAME                = 'algopy'
MAINTAINER          = "Sebastian F. Walter"
MAINTAINER_EMAIL    = "sebastian.walter@gmail.com"
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://www.github.com/b45ch1/algopy"
DOWNLOAD_URL        = "http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103"
LICENSE             = 'BSD'
CLASSIFIERS         = filter(None, CLASSIFIERS.split('\n'))
AUTHOR              = "Sebastian F. Walter"
AUTHOR_EMAIL        = "sebastian.walter@gmail.com"
PLATFORMS           = ["Linux"]
MAJOR               = 0
MINOR               = 1 
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    # If in git or something, bypass the svn rev
    if os.path.exists('.svn'):
        FULLVERSION += svn_version()
        
# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def write_version_py(filename='algopy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM ALGOPY SETUP.PY
short_version='%(version)s'
version='%(version)s'
release=%(isrelease)s

if not release:
    version += '.dev'
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()

def fullsplit(path, result=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """
    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)

def setup_package():

    # Rewrite the version file everytime
    if os.path.exists('algopy/version.py'): os.remove('algopy/version.py')
    write_version_py()

    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)
    
    # find all files that should be included
    packages, data_files = [], []
    for dirpath, dirnames, filenames in os.walk('algopy'):
        # Ignore dirnames that start with '.'
        for i, dirname in enumerate(dirnames):
            if dirname.startswith('.'): del dirnames[i]
        if '__init__.py' in filenames:
            packages.append('.'.join(fullsplit(dirpath)))
        elif filenames:
            data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])

    from distutils.core import setup
    
    try:
        setup(name=NAME,
          version=VERSION,
          description = DESCRIPTION,
          long_description = LONG_DESCRIPTION,
          license=LICENSE,
          author=AUTHOR,
          platforms=PLATFORMS,
          author_email= AUTHOR_EMAIL,
          url=URL,
          packages = packages,
         )

    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
