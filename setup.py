#!/usr/bin/env python
"""ALGOPY: Taylor Arithmetic Computation and Algorithmic Differentiation

ALGOPY is a tool for Algorithmic Differentiation (AD) and Taylor polynomial approximations.
ALGOPY makes it possible to perform computations on scalar and polynomial matrices.
It is designed to be as compatible to numpy as possible. I.e. views, broadcasting and most
functions of numpy can be performed on polynomial matrices. Exampels are dot,trace,qr,solve,
inv,eigh.
The reverse mode of AD is also supported by a simple code evaluation tracer.

Documentation with examples is available at http://packages.python.org/algopy/.

"""

#Upload to pypi::
#
#    python -m build --sdist
#    twine upload dist/*

# upload sphinx documentation
#    python setup.py build_sphinx
#    python setup.py upload_sphinx
# need to uncomment some code for that (look for comments containing build_sphinx)

DOCLINES = __doc__.split("\n")

import os
import shutil
import sys
import re
import subprocess

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Development Status :: 4 - Beta
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
"""

NAME                = 'algopy'
MAINTAINER          = "Sebastian F. Walter"
MAINTAINER_EMAIL    = "sebastian.walter@gmail.com"
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
KEYWORDS            = ['algorithmic differentiation', 'computational differentiation', 'automatic differentiation', 'forward mode', 'reverse mode', 'Taylor arithmetic']
URL                 = "https://packages.python.org/algopy"
DOWNLOAD_URL        = "https://www.github.com/b45ch1/algopy"
LICENSE             = 'BSD'
CLASSIFIERS         = [_f for _f in CLASSIFIERS.split('\n') if _f]
AUTHOR              = "Sebastian F. Walter"
AUTHOR_EMAIL        = "sebastian.walter@gmail.com"
PLATFORMS           = ["all"]
MAJOR               = 0
MINOR               = 7
MICRO               = 2
ISRELEASED          = True
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def write_version_py(filename='algopy/version.py'):
    try:
        gitfile = open('.git/refs/heads/master', 'r')
        git_revision = '.dev' + gitfile.readline().split('\n')[0]

    except:
        git_revision = '.dev'

    print(git_revision)

    cnt = """
# THIS FILE IS GENERATED FROM ALGOPY SETUP.PY
short_version='%(version)s'
version='%(version)s'
release=%(isrelease)s

if not release:
    version += '%(git_revision)s'
"""


    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'isrelease': str(ISRELEASED), 'git_revision':git_revision})
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

    import pathlib
    local_path = pathlib.Path(__file__).parent.resolve()

    print('local_path', local_path)
    src_path = local_path

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # find all files that should be included
    packages, data_files = [], []
    for dirpath, dirnames, filenames in os.walk('algopy'):
        print(dirpath)
        # Ignore dirnames that start with '.'
        for i, dirname in enumerate(dirnames):
            if dirname.startswith('.'): del dirnames[i]
        if '__init__.py' in filenames:
            packages.append('.'.join(fullsplit(dirpath)))
        elif filenames:
            data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])

    from setuptools import setup #uncomment for build_sphinx and upload_sphinx


    print('packages', packages)

    try:
        setup(name=NAME,
          version=VERSION,
          description = DESCRIPTION,
          long_description = LONG_DESCRIPTION,
          license=LICENSE,
          author=AUTHOR,
          platforms=PLATFORMS,
          author_email= AUTHOR_EMAIL,
          keywords = KEYWORDS,
          url=URL,
          packages = packages,
          # ext_package='algopy.ctps',
          # ext_modules=[Extension('libctps', ['algopy/ctps/src/ctps.c'])],
          # entry_points = {"distutils.commands": ["upload_sphinx = sphinx_pypi_upload:UploadDoc",]} #uncomment for build_sphinx and upload_sphinx
         )

    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
