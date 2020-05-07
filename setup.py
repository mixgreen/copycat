#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

import versioneer

if sys.version_info[:3] < (3, 5, 3):
    raise Exception("This software requires Python 3.5.3+")

requirements = [
    # "artiq",  # Needs to be installed manually (see ARTIQ installation instructions)
    "numpy",
    "scipy",
    "pyvcd",
    "natsort",
    "pygit2",
]

setup(
    name="dax",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Duke University",
    author_email="",
    description="Duke ARTIQ Extensions",
    install_requires=requirements,
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
