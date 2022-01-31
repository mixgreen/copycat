import dax.base.system
import dax.modules.safety_context
from dax.modules.safety_context import SafetyContextError

__all__ = ['ReentrantSafetyContextService', 'SafetyContextService', 'SafetyContextError']


class ReentrantSafetyContextService(dax.modules.safety_context.BaseReentrantSafetyContext, dax.base.system.DaxService):
    """Context class for a service for safety controls when entering and exiting a context."""
    pass


class SafetyContextService(dax.modules.safety_context.BaseNonReentrantSafetyContext, dax.base.system.DaxService):
    """Context class for a service for safety controls when entering and exiting a context."""
    pass
