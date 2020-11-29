import os
import distutils.util
import random

__all__ = ['CI_ENABLED', 'TB_DISABLED', 'JOB_ID']

CI_ENABLED = distutils.util.strtobool(os.getenv('GITLAB_CI', '0'))  # Only GitLab CI
"""True if tests are running in a CI environment."""

TB_DISABLED = distutils.util.strtobool(os.getenv('TB_DISABLED', '0'))
"""True if hardware testbenches are disabled."""

JOB_ID = os.getenv('CI_JOB_ID', f'0{random.randrange(2 ** 32)}')  # Random ID does not collide with any job ID
"""The unique ID of the current CI job or a random ID of none is available."""
