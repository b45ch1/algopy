#!/usr/bin/env python
"""ALGOPY: Taylor arithmetic computation and Algorithmic Differentiation
in the forward and reverse mode of Scalars and Matrices
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


def setup_package():

    # Rewrite the version file everytime
    if os.path.exists('algopy/version.py'): os.remove('algopy/version.py')
    write_version_py()

    # Perform 2to3 if needed
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    from distutils.core import setup
    
    try:
        setup(name=NAME,
          version=VERSION,
          description = DESCRIPTION,
          long_description = DESCRIPTION,
          license=LICENSE,
          author=AUTHOR,
          platforms=PLATFORMS,
          author_email= AUTHOR_EMAIL,
          url=URL,
          packages=['algopy']
         )

    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
