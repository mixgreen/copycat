#!/usr/bin/env python3

from setuptools import setup
import sys

import versioneer

if sys.version_info[:3] < (3, 5, 3):
    raise Exception("This software requires Python 3.5.3+")

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
