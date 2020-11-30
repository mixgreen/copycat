import os
import distutils.util
import random

__all__ = ['NIX_ENV', 'CONDA_ENV', 'CI_ENABLED', 'JOB_ID', 'TB_DISABLED']

NIX_ENV: bool = bool(os.getenv('NIX_STORE'))
"""True if we are in a Nix environment."""

CONDA_ENV: bool = bool(os.getenv('CONDA_DEFAULT_ENV'))
"""True if we are in a Conda environment."""

CI_ENABLED: bool = distutils.util.strtobool(os.getenv('GITLAB_CI', '0'))  # Only GitLab CI
"""True if we are running in a CI environment."""

JOB_ID: str = os.getenv('CI_JOB_ID', f'0{random.randrange(2 ** 32)}')  # Random ID does not collide with any job ID
"""The unique ID of the current CI job or a random ID of none is available."""

TB_DISABLED: bool = distutils.util.strtobool(os.getenv('TB_DISABLED', '0'))
"""True if hardware testbenches are disabled."""
