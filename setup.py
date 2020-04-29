#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

if sys.version_info[:3] < (3, 5, 3):
    raise Exception("This software requires Python 3.5.3+")

requirements = [
    # "artiq",  # Needs to be installed manually (see ARTIQ installation instructions)
    "numpy",
    "scipy",
    "pyvcd",
    "natsort",
    "mypy",
    "gitpython",
    "pycodestyle",
]

setup(
    name="dax",
    version="0.1",
    author="Duke University",
    author_email="",
    description="Duke ARTIQ Extensions",
    install_requires=requirements,
    packages=find_packages(),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
