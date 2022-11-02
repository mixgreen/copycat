import unittest

from dax.sim.coredevice.overlay import Overlay


class _Obj:
    def __init__(self, val=0):
        self._val = val

    def get_val(self):
        return self._val

    def set_val(self, val):
        self._val = val


class _Overlay(Overlay):
    def set_val(self, val):
        self._obj.set_val(val)
        self._parent.set_val(val)


class OverlayTestCase(unittest.TestCase):
    def test_overlay_passthrough(self):
        obj = _Obj(0)
        self.assertEqual(obj._val, 0)
        ol = Overlay(self, obj)
        self.assertEqual(ol.get_val(), 0)
        ol.set_val(3)
        self.assertEqual(ol.get_val(), 3)
        self.assertEqual(obj.get_val(), 3)

    def test_overlay_parent(self):
        obj = _Obj(0)
        self.assertEqual(obj._val, 0)
        parent = _Obj(0)
        ol = _Overlay(parent, obj)
        self.assertEqual(parent.get_val(), 0)
        self.assertEqual(ol.get_val(), 0)
        ol.set_val(3)
        self.assertEqual(parent.get_val(), 3)
        self.assertEqual(ol.get_val(), 3)
        self.assertEqual(obj.get_val(), 3)
