#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import os.path as op
from setuptools import setup, find_packages

execfile(op.join('cardo','version.py')) #get __version__

here = op.abspath(op.dirname(__file__))

# Get the long description from the README file
with open(op.join(here, 'README.md')) as f:
    long_description = f.read()

setup(name='cardo', version=__version__,
      description='Make table reporting folders organized as a UB-tree',
      long_description=long_description, author='Thomas Vincent', license='MIT',
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Topic ::  :: Build Tools',
                   'Environment :: Console',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Multimedia :: Graphics :: Presentation',
                   'Topic :: Scientific/Engineering :: Visualization'],
      keywords='cardo reporting SVG',
      packages=find_packages(exclude=['misc', 'test']),
      install_requires=['svgwrite', 'numpy'],
      entry_points={
          'console_scripts': [
              'cardo = cardo.commands.__main__:main',
          ],
      })
