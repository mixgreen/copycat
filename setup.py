#!/usr/bin/env python3

from setuptools import setup
import sys

import versioneer

if sys.version_info[:2] < (3, 7):
    raise Exception("This software requires Python 3.7+")

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
