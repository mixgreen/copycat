import os

__all__ = ['CI_ENABLED']

# Evaluates True in a CI environment
CI_ENABLED = os.getenv('CI')
