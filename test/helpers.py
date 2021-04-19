"""File with test helper functions."""


def test_system_kernel_invariants(test_case, system):
    """Test all kernel invariants of a system.

    :param test_case: The test case class
    :param system: The DAX system
    """
    for m in system.registry.get_module_list():
        test_kernel_invariants(test_case, m)
    for s in system.registry.get_service_list():
        test_kernel_invariants(test_case, s)


def test_kernel_invariants(test_case, component):
    """Test kernel invariants of a component.

    :param test_case: The test case class
    :param component: The component to test
    """
    try:
        component_name = component.get_system_key()
    except AttributeError:
        component_name = component

    for k in getattr(component, 'kernel_invariants', set()):
        test_case.assertTrue(hasattr(component, k), f'Name "{k}" of "{component_name}" was marked '
                                                    f'kernel invariant, but this attribute does not exist')
