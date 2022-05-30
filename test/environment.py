import os
import distutils.util

__all__ = ['NIX_ENV', 'CONDA_ENV', 'CI_ENABLED', 'HW_TEST_ENABLED']

NIX_ENV: bool = bool(os.getenv('NIX_STORE')) or 'nixos/nix' in os.getenv('CI_JOB_IMAGE', '')
"""True if we are in a Nix shell or in a CI environment with a ``nixos/nix`` docker image."""

CONDA_ENV: bool = bool(os.getenv('CONDA_DEFAULT_ENV'))
"""True if we are in a Conda environment."""

CI_ENABLED: bool = bool(distutils.util.strtobool(os.getenv('GITLAB_CI', '0')))  # Only GitLab CI
"""True if we are running in a CI environment."""

HW_TEST_ENABLED: bool = bool(distutils.util.strtobool(os.getenv('HW_TEST_ENABLED', '0')))
"""True if the hardware test flag is set."""
