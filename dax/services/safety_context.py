from dax.experiment import DaxService
from dax.modules.safety_context import SafetyContextError
import dax.modules.safety_context

__all__ = ['ReentrantSafetyContextService', 'SafetyContextService', 'SafetyContextError']


class ReentrantSafetyContextService(dax.modules.safety_context.BaseReentrantSafetyContext, DaxService):
    """Context class for a service for safety controls when entering and exiting a context"""
    SERVICE_NAME = 'reentrant_safety_context'


class SafetyContextService(dax.modules.safety_context.BaseNonReentrantSafetyContext, DaxService):
    """Context class for a service for safety controls when entering and exiting a non reentrant context"""
    SERVICE_NAME = 'nonreentrant_safety_context'
