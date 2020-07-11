# flake8: noqa
from .attributes import *
from .code_style import *

# Suppress logging for all tests if test discovery is used
import logging

logging.basicConfig(level=logging.CRITICAL)
