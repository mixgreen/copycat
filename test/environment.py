import os

__all__ = ['CI_ENABLED', 'TB_DISABLED']

CI_ENABLED = os.getenv('CI')
"""Evaluates true if tests are running in a CI environment."""

TB_DISABLED = os.getenv('TB_DISABLED')
"""Evaluates true if hardware testbenches are disabled."""
