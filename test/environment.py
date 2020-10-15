import os

__all__ = ['CI_ENABLED']

# Evaluates True in the GitLab CI environment
CI_ENABLED = os.getenv('GITLAB_CI')
