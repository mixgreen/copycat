#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys


if sys.version_info[:3] < (3, 5, 3):
    raise Exception("You need Python 3.5.3+")


# Depends on PyQt5, but setuptools cannot check for it.
requirements = [
    # "artiq", # needs to be installed manually (see ARTIQ installation instructions)
    "h5py",
    "numpy",
    # "ok", # required, but needs to be installed manually (OpalKelly FrontPanel)
    "pyqtgraph",
    "pyvisa",
    "pyvisa-py",
    "scipy"
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
    long_description_content_type="text/markdown"
)
