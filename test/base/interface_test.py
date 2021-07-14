import unittest

import dax.base.interface

from dax.base.interface import optional, is_optional, get_optionals


class _TestClass(dax.base.interface.DaxInterface):
    @optional
    def optional_method(self):
        pass

    def normal_method(self):
        pass

    @classmethod
    def class_method(cls):
        pass

    @staticmethod
    def static_method():
        pass


class InterfaceTestCase(unittest.TestCase):
    _OPTIONAL_METHODS = {'optional_method'}
    _OTHER_METHODS = {'normal_method', 'class_method', 'static_method'}

    def test_optional_decorator(self):
        for obj in [_TestClass, _TestClass()]:
            for fn_name in self._OPTIONAL_METHODS:
                self.assertTrue(getattr(getattr(obj, fn_name), dax.base.interface._OPTIONAL_METHOD_KEY, False))
            for fn_name in self._OTHER_METHODS:
                self.assertFalse(getattr(getattr(obj, fn_name), dax.base.interface._OPTIONAL_METHOD_KEY, False))

    def test_is_optional(self):
        for obj in [_TestClass, _TestClass()]:
            for fn_name in self._OPTIONAL_METHODS:
                self.assertTrue(is_optional(getattr(obj, fn_name)))
            for fn_name in self._OTHER_METHODS:
                self.assertFalse(is_optional(getattr(obj, fn_name)))

    def test_is_optional_none(self):
        self.assertFalse(is_optional(None))  # This is allowed

    def test_get_optionals(self):
        for obj in [_TestClass, _TestClass()]:
            optionals = get_optionals(obj)
            self.assertSetEqual(optionals, self._OPTIONAL_METHODS)
            self.assertSetEqual(optionals & self._OTHER_METHODS, set())
